"""Dense float64 HMC kernels exposed through a C ABI."""

from std.ffi import c_int, external_call
from std.math import sqrt
from std.runtime import initialize_runtime
from std.runtime.asyncrt import create_task
from std.sys.info import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime ELEMENT_WORKERS = 8


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


@export("mpmc_create_cpu_context")
def mpmc_create_cpu_context() abi("C") -> Int:
    # Shared libraries do not run Mojo's normal startup path.  The 1.1 runtime
    # is process-global and remains alive for the process lifetime, so preserve
    # the Python/C lifecycle ABI with a stable non-null opaque token.
    initialize_runtime()
    return 1


@export("mpmc_destroy_cpu_context")
def mpmc_destroy_cpu_context(ctx_addr: Int) abi("C"):
    pass


def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    var vacc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        vacc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var acc = vacc.reduce_add()
    while i < n:
        acc += a[i] * b[i]
        i += 1
    return acc


def diag_velocity_energy(x: Ptr, diag: Ptr, velocity: Ptr, n: Int) -> Float64:
    var vacc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        var vx = x.load[width=W](i)
        var vv = vx * diag.load[width=W](i)
        velocity.store(i, vv)
        vacc += vx * vv
        i += W
    var energy = vacc.reduce_add()
    while i < n:
        var v = x[i] * diag[i]
        velocity[i] = v
        energy += x[i] * v
        i += 1
    return 0.5 * energy


async def diag_velocity_energy_chunk(
    x: Ptr,
    diag: Ptr,
    velocity: Ptr,
    scratch: Ptr,
    n: Int,
    chunk_size: Int,
    task: Int,
):
    var start = task * chunk_size
    var end = min(start + chunk_size, n)
    var vacc = SIMD[DType.float64, W](0.0)
    var i = start
    while i + W <= end:
        var vx = x.load[width=W](i)
        var vv = vx * diag.load[width=W](i)
        velocity.store(i, vv)
        vacc += vx * vv
        i += W
    var energy = vacc.reduce_add()
    while i < end:
        var v = x[i] * diag[i]
        velocity[i] = v
        energy += x[i] * v
        i += 1
    scratch[task] = energy


def diag_velocity_energy_parallel(
    x: Ptr,
    diag: Ptr,
    velocity: Ptr,
    scratch: Ptr,
    n: Int,
) -> Float64:
    var chunk_size = (
        ((n + ELEMENT_WORKERS - 1) // ELEMENT_WORKERS + W - 1) // W * W
    )

    var task0 = create_task(
        diag_velocity_energy_chunk(x, diag, velocity, scratch, n, chunk_size, 0)
    )
    var task1 = create_task(
        diag_velocity_energy_chunk(x, diag, velocity, scratch, n, chunk_size, 1)
    )
    var task2 = create_task(
        diag_velocity_energy_chunk(x, diag, velocity, scratch, n, chunk_size, 2)
    )
    var task3 = create_task(
        diag_velocity_energy_chunk(x, diag, velocity, scratch, n, chunk_size, 3)
    )
    var task4 = create_task(
        diag_velocity_energy_chunk(x, diag, velocity, scratch, n, chunk_size, 4)
    )
    var task5 = create_task(
        diag_velocity_energy_chunk(x, diag, velocity, scratch, n, chunk_size, 5)
    )
    var task6 = create_task(
        diag_velocity_energy_chunk(x, diag, velocity, scratch, n, chunk_size, 6)
    )
    var task7 = create_task(
        diag_velocity_energy_chunk(x, diag, velocity, scratch, n, chunk_size, 7)
    )
    _ = task0.wait()
    _ = task1.wait()
    _ = task2.wait()
    _ = task3.wait()
    _ = task4.wait()
    _ = task5.wait()
    _ = task6.wait()
    _ = task7.wait()
    var energy = 0.0
    for task in range(ELEMENT_WORKERS):
        energy += scratch[task]
    return 0.5 * energy


def full_velocity(x: Ptr, matrix: Ptr, velocity: Ptr, n: Int):
    external_call["cblas_dgemv", NoneType](
        c_int(101),
        c_int(111),
        c_int(n),
        c_int(n),
        1.0,
        matrix,
        c_int(n),
        x,
        c_int(1),
        0.0,
        velocity,
        c_int(1),
    )


def full_velocity_energy(x: Ptr, matrix: Ptr, velocity: Ptr, n: Int) -> Float64:
    full_velocity(x, matrix, velocity, n)
    return 0.5 * external_call["cblas_ddot", Float64](
        c_int(n), x, c_int(1), velocity, c_int(1)
    )


def chol_solve(chol: Ptr, rhs: Ptr, solution: Ptr, n: Int):
    for i in range(n):
        var acc = rhs[i]
        for k in range(i):
            acc -= chol[i * n + k] * solution[k]
        solution[i] = acc / chol[i * n + i]
    for ri in range(n):
        var i = n - 1 - ri
        var acc = solution[i]
        for k in range(i + 1, n):
            acc -= chol[k * n + i] * solution[k]
        solution[i] = acc / chol[i * n + i]


def inv_velocity_energy(x: Ptr, chol: Ptr, velocity: Ptr, n: Int) -> Float64:
    chol_solve(chol, x, velocity, n)
    return 0.5 * dot(x, velocity, n)


def add_scaled(y: Ptr, x: Ptr, alpha: Float64, n: Int):
    var va = SIMD[DType.float64, W](alpha)
    var i = 0
    while i + W <= n:
        y.store(i, y.load[width=W](i) + va * x.load[width=W](i))
        i += W
    while i < n:
        y[i] += alpha * x[i]
        i += 1


def leapfrog_first_diag(
    q: Ptr,
    momentum: Ptr,
    grad: Ptr,
    diag: Ptr,
    velocity: Ptr,
    epsilon: Float64,
    n: Int,
):
    var dt = 0.5 * epsilon
    var veps = SIMD[DType.float64, W](epsilon)
    var vdt = SIMD[DType.float64, W](dt)
    var i = 0
    while i + W <= n:
        var vp = momentum.load[width=W](i) + vdt * grad.load[width=W](i)
        var vv = vp * diag.load[width=W](i)
        momentum.store(i, vp)
        velocity.store(i, vv)
        q.store(i, q.load[width=W](i) + veps * vv)
        i += W
    while i < n:
        momentum[i] += dt * grad[i]
        velocity[i] = momentum[i] * diag[i]
        q[i] += epsilon * velocity[i]
        i += 1


async def leapfrog_first_diag_chunk(
    q: Ptr,
    momentum: Ptr,
    grad: Ptr,
    diag: Ptr,
    velocity: Ptr,
    epsilon: Float64,
    n: Int,
    chunk_size: Int,
    task: Int,
):
    var start = task * chunk_size
    var end = min(start + chunk_size, n)
    var dt = 0.5 * epsilon
    var veps = SIMD[DType.float64, W](epsilon)
    var vdt = SIMD[DType.float64, W](dt)
    var i = start
    while i + W <= end:
        var vp = momentum.load[width=W](i) + vdt * grad.load[width=W](i)
        var vv = vp * diag.load[width=W](i)
        momentum.store(i, vp)
        velocity.store(i, vv)
        q.store(i, q.load[width=W](i) + veps * vv)
        i += W
    while i < end:
        momentum[i] += dt * grad[i]
        velocity[i] = momentum[i] * diag[i]
        q[i] += epsilon * velocity[i]
        i += 1


def leapfrog_first_diag_parallel(
    q: Ptr,
    momentum: Ptr,
    grad: Ptr,
    diag: Ptr,
    velocity: Ptr,
    epsilon: Float64,
    n: Int,
):
    var chunk_size = (
        ((n + ELEMENT_WORKERS - 1) // ELEMENT_WORKERS + W - 1) // W * W
    )

    var task0 = create_task(
        leapfrog_first_diag_chunk(
            q, momentum, grad, diag, velocity, epsilon, n, chunk_size, 0
        )
    )
    var task1 = create_task(
        leapfrog_first_diag_chunk(
            q, momentum, grad, diag, velocity, epsilon, n, chunk_size, 1
        )
    )
    var task2 = create_task(
        leapfrog_first_diag_chunk(
            q, momentum, grad, diag, velocity, epsilon, n, chunk_size, 2
        )
    )
    var task3 = create_task(
        leapfrog_first_diag_chunk(
            q, momentum, grad, diag, velocity, epsilon, n, chunk_size, 3
        )
    )
    var task4 = create_task(
        leapfrog_first_diag_chunk(
            q, momentum, grad, diag, velocity, epsilon, n, chunk_size, 4
        )
    )
    var task5 = create_task(
        leapfrog_first_diag_chunk(
            q, momentum, grad, diag, velocity, epsilon, n, chunk_size, 5
        )
    )
    var task6 = create_task(
        leapfrog_first_diag_chunk(
            q, momentum, grad, diag, velocity, epsilon, n, chunk_size, 6
        )
    )
    var task7 = create_task(
        leapfrog_first_diag_chunk(
            q, momentum, grad, diag, velocity, epsilon, n, chunk_size, 7
        )
    )
    _ = task0.wait()
    _ = task1.wait()
    _ = task2.wait()
    _ = task3.wait()
    _ = task4.wait()
    _ = task5.wait()
    _ = task6.wait()
    _ = task7.wait()


def leapfrog_first_full(
    q: Ptr,
    momentum: Ptr,
    grad: Ptr,
    matrix: Ptr,
    velocity: Ptr,
    epsilon: Float64,
    n: Int,
):
    add_scaled(momentum, grad, 0.5 * epsilon, n)
    full_velocity(momentum, matrix, velocity, n)
    add_scaled(q, velocity, epsilon, n)


def leapfrog_first_inv(
    q: Ptr,
    momentum: Ptr,
    grad: Ptr,
    chol: Ptr,
    velocity: Ptr,
    epsilon: Float64,
    n: Int,
):
    add_scaled(momentum, grad, 0.5 * epsilon, n)
    chol_solve(chol, momentum, velocity, n)
    add_scaled(q, velocity, epsilon, n)


def welford_var_add(
    value: Ptr, mean: Ptr, raw_var: Ptr, weight: Float64, n: Int
):
    for i in range(n):
        var old_diff = value[i] - mean[i]
        mean[i] += old_diff / weight
        raw_var[i] += old_diff * (value[i] - mean[i])


def exp_var_add(value: Ptr, mean: Ptr, variance: Ptr, alpha: Float64, n: Int):
    for i in range(n):
        var delta = value[i] - mean[i]
        mean[i] += alpha * delta
        variance[i] = (1.0 - alpha) * (variance[i] + alpha * delta * delta)


def welford_cov_add(
    value: Ptr,
    mean: Ptr,
    raw_cov: Ptr,
    old_diff: Ptr,
    new_diff: Ptr,
    weight: Float64,
    n: Int,
):
    for i in range(n):
        old_diff[i] = value[i] - mean[i]
        mean[i] += old_diff[i] / weight
        new_diff[i] = value[i] - mean[i]
    for i in range(n):
        var nd = new_diff[i]
        for j in range(n):
            raw_cov[i * n + j] += nd * old_diff[j]


@export("mpmc_diag_velocity_energy")
def mpmc_diag_velocity_energy(
    x: Int, diag: Int, velocity: Int, n: Int
) abi("C") -> Float64:
    return diag_velocity_energy(p(x), p(diag), p(velocity), n)


@export("mpmc_diag_velocity_energy_parallel")
def mpmc_diag_velocity_energy_parallel(
    ctx: Int, x: Int, diag: Int, velocity: Int, scratch: Int, n: Int
) abi("C") -> Float64:
    return diag_velocity_energy_parallel(
        p(x),
        p(diag),
        p(velocity),
        p(scratch),
        n,
    )


@export("mpmc_full_velocity_energy")
def mpmc_full_velocity_energy(
    x: Int, matrix: Int, velocity: Int, n: Int
) abi("C") -> Float64:
    return full_velocity_energy(p(x), p(matrix), p(velocity), n)


@export("mpmc_full_velocity")
def mpmc_full_velocity(x: Int, matrix: Int, velocity: Int, n: Int) abi("C"):
    full_velocity(p(x), p(matrix), p(velocity), n)


@export("mpmc_inv_velocity_energy")
def mpmc_inv_velocity_energy(
    x: Int, chol: Int, velocity: Int, n: Int
) abi("C") -> Float64:
    return inv_velocity_energy(p(x), p(chol), p(velocity), n)


@export("mpmc_add_scaled")
def mpmc_add_scaled(y: Int, x: Int, alpha: Float64, n: Int) abi("C"):
    add_scaled(p(y), p(x), alpha, n)


@export("mpmc_leapfrog_first_diag")
def mpmc_leapfrog_first_diag(
    q: Int,
    momentum: Int,
    grad: Int,
    diag: Int,
    velocity: Int,
    epsilon: Float64,
    n: Int,
) abi("C"):
    leapfrog_first_diag(
        p(q), p(momentum), p(grad), p(diag), p(velocity), epsilon, n
    )


@export("mpmc_leapfrog_first_diag_parallel")
def mpmc_leapfrog_first_diag_parallel(
    ctx: Int,
    q: Int,
    momentum: Int,
    grad: Int,
    diag: Int,
    velocity: Int,
    epsilon: Float64,
    n: Int,
) abi("C"):
    leapfrog_first_diag_parallel(
        p(q),
        p(momentum),
        p(grad),
        p(diag),
        p(velocity),
        epsilon,
        n,
    )


@export("mpmc_leapfrog_first_full")
def mpmc_leapfrog_first_full(
    q: Int,
    momentum: Int,
    grad: Int,
    matrix: Int,
    velocity: Int,
    epsilon: Float64,
    n: Int,
) abi("C"):
    leapfrog_first_full(
        p(q), p(momentum), p(grad), p(matrix), p(velocity), epsilon, n
    )


@export("mpmc_leapfrog_first_inv")
def mpmc_leapfrog_first_inv(
    q: Int,
    momentum: Int,
    grad: Int,
    chol: Int,
    velocity: Int,
    epsilon: Float64,
    n: Int,
) abi("C"):
    leapfrog_first_inv(
        p(q), p(momentum), p(grad), p(chol), p(velocity), epsilon, n
    )


@export("mpmc_welford_var_add")
def mpmc_welford_var_add(
    value: Int, mean: Int, raw_var: Int, weight: Float64, n: Int
) abi("C"):
    welford_var_add(p(value), p(mean), p(raw_var), weight, n)


@export("mpmc_exp_var_add")
def mpmc_exp_var_add(
    value: Int, mean: Int, variance: Int, alpha: Float64, n: Int
) abi("C"):
    exp_var_add(p(value), p(mean), p(variance), alpha, n)


@export("mpmc_welford_cov_add")
def mpmc_welford_cov_add(
    value: Int,
    mean: Int,
    raw_cov: Int,
    old_diff: Int,
    new_diff: Int,
    weight: Float64,
    n: Int,
) abi("C"):
    welford_cov_add(
        p(value), p(mean), p(raw_cov), p(old_diff), p(new_diff), weight, n
    )


@export("mpmc_is_turning")
def mpmc_is_turning(
    momentum_sum: Int, left_velocity: Int, right_velocity: Int, n: Int
) abi("C") -> Int:
    var momentum = p(momentum_sum)
    var left = p(left_velocity)
    var right = p(right_velocity)
    if n >= 8_192:
        var left_dot = external_call["cblas_ddot", Float64](
            c_int(n), momentum, c_int(1), left, c_int(1)
        )
        var right_dot = external_call["cblas_ddot", Float64](
            c_int(n), momentum, c_int(1), right, c_int(1)
        )
        return 1 if left_dot <= 0.0 or right_dot <= 0.0 else 0
    var left0 = SIMD[DType.float64, W](0.0)
    var left1 = SIMD[DType.float64, W](0.0)
    var left2 = SIMD[DType.float64, W](0.0)
    var left3 = SIMD[DType.float64, W](0.0)
    var right0 = SIMD[DType.float64, W](0.0)
    var right1 = SIMD[DType.float64, W](0.0)
    var right2 = SIMD[DType.float64, W](0.0)
    var right3 = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + 4 * W <= n:
        var momentum0 = momentum.load[width=W](i)
        var momentum1 = momentum.load[width=W](i + W)
        var momentum2 = momentum.load[width=W](i + 2 * W)
        var momentum3 = momentum.load[width=W](i + 3 * W)
        left0 += momentum0 * left.load[width=W](i)
        left1 += momentum1 * left.load[width=W](i + W)
        left2 += momentum2 * left.load[width=W](i + 2 * W)
        left3 += momentum3 * left.load[width=W](i + 3 * W)
        right0 += momentum0 * right.load[width=W](i)
        right1 += momentum1 * right.load[width=W](i + W)
        right2 += momentum2 * right.load[width=W](i + 2 * W)
        right3 += momentum3 * right.load[width=W](i + 3 * W)
        i += 4 * W
    while i + W <= n:
        var momentum_v = momentum.load[width=W](i)
        left0 += momentum_v * left.load[width=W](i)
        right0 += momentum_v * right.load[width=W](i)
        i += W
    var left_dot = (left0 + left1 + left2 + left3).reduce_add()
    var right_dot = (right0 + right1 + right2 + right3).reduce_add()
    while i < n:
        left_dot += momentum[i] * left[i]
        right_dot += momentum[i] * right[i]
        i += 1
    return 1 if left_dot <= 0.0 or right_dot <= 0.0 else 0
