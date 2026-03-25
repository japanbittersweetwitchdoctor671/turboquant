const std = @import("std");

pub const PolarError = error{ InvalidDimension, OutOfMemory };

const R_BITS: u5 = 4;
const THETA_BITS: u5 = 3;
const BITS_PER_PAIR: u5 = R_BITS + THETA_BITS;
const R_LEVELS: f32 = 15.0;
const THETA_LEVELS: f32 = 7.0;

const ANGLE_BUCKETS: usize = 8;
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 2.0 * PI;

const polar_sin_table: [ANGLE_BUCKETS]f32 = init: {
    var table: [ANGLE_BUCKETS]f32 = undefined;
    for (0..ANGLE_BUCKETS) |i| {
        const theta = @as(f32, @floatFromInt(i)) / THETA_LEVELS * TWO_PI - PI;
        table[i] = @sin(theta);
    }
    break :init table;
};

const polar_cos_table: [ANGLE_BUCKETS]f32 = init: {
    var table: [ANGLE_BUCKETS]f32 = undefined;
    for (0..ANGLE_BUCKETS) |i| {
        const theta = @as(f32, @floatFromInt(i)) / THETA_LEVELS * TWO_PI - PI;
        table[i] = @cos(theta);
    }
    break :init table;
};

const direction_vectors: [ANGLE_BUCKETS][2]f32 = init: {
    var dirs: [ANGLE_BUCKETS][2]f32 = undefined;
    for (0..ANGLE_BUCKETS) |i| {
        const theta = @as(f32, @floatFromInt(i)) / THETA_LEVELS * TWO_PI - PI;
        dirs[i] = .{ @cos(theta), @sin(theta) };
    }
    break :init dirs;
};

pub fn cosTable() *const [8]f32 {
    return &polar_cos_table;
}

pub fn sinTable() *const [8]f32 {
    return &polar_sin_table;
}

fn findNearestAngleBucket(x: f32, y: f32) u3 {
    var best_bucket: u3 = 0;
    var best_dot: f32 = -1.0;
    for (0..ANGLE_BUCKETS) |i| {
        const dot = x * direction_vectors[i][0] + y * direction_vectors[i][1];
        if (dot > best_dot) {
            best_dot = dot;
            best_bucket = @intCast(i);
        }
    }
    return best_bucket;
}

pub fn encode(
    allocator: std.mem.Allocator,
    rotated: []const f32,
    max_r: f32,
) PolarError![]u8 {
    const dim = rotated.len;
    if (dim == 0 or dim % 2 != 0) return PolarError.InvalidDimension;

    const num_pairs = dim / 2;
    const polar_bits = num_pairs * BITS_PER_PAIR;
    const polar_bytes = (polar_bits + 7) / 8;

    const result = try allocator.alloc(u8, polar_bytes);
    @memset(result, 0);

    var bit_pos: usize = 0;
    for (0..num_pairs) |i| {
        const x = rotated[i * 2];
        const y = rotated[i * 2 + 1];
        const r = @sqrt(x * x + y * y);

        const r_quant = @as(u4, @intFromFloat(r / max_r * R_LEVELS));
        const theta_bucket = findNearestAngleBucket(x, y);
        const combined: u7 = (@as(u7, r_quant) << THETA_BITS) | theta_bucket;

        for (0..BITS_PER_PAIR) |j| {
            const bit = (combined >> @intCast(BITS_PER_PAIR - 1 - j)) & 1;
            if (bit == 1) {
                result[bit_pos / 8] |= @as(u8, 1) << @intCast(bit_pos % 8);
            }
            bit_pos += 1;
        }
    }

    return result;
}

pub fn decode(
    allocator: std.mem.Allocator,
    compressed: []const u8,
    dim: usize,
    max_r: f32,
) PolarError![]f32 {
    if (dim == 0 or dim % 2 != 0) return PolarError.InvalidDimension;

    const num_pairs = dim / 2;
    const result = try allocator.alloc(f32, dim);
    errdefer allocator.free(result);

    var bit_pos: usize = 0;
    for (0..num_pairs) |i| {
        var combined: u7 = 0;
        for (0..BITS_PER_PAIR) |j| {
            const bit: u7 = @intCast((compressed[(bit_pos + j) / 8] >> @intCast((bit_pos + j) % 8)) & 1);
            combined = (combined << 1) | bit;
        }
        bit_pos += BITS_PER_PAIR;

        const r = @as(f32, @floatFromInt((combined >> THETA_BITS) & 0xF)) / R_LEVELS * max_r;
        const bucket = @as(u3, @intCast(combined & 0x7));

        result[i * 2] = r * polar_cos_table[bucket];
        result[i * 2 + 1] = r * polar_sin_table[bucket];
    }

    return result;
}

pub fn dotProduct(
    q: []const f32,
    compressed: []const u8,
    max_r: f32,
) f32 {
    const dim = q.len;
    if (dim == 0 or dim % 2 != 0) return 0;

    const num_pairs = dim / 2;
    var sum: f32 = 0;
    var bit_pos: usize = 0;

    for (0..num_pairs) |i| {
        var combined: u7 = 0;
        for (0..BITS_PER_PAIR) |j| {
            const bit: u7 = @intCast((compressed[(bit_pos + j) / 8] >> @intCast((bit_pos + j) % 8)) & 1);
            combined = (combined << 1) | bit;
        }
        bit_pos += BITS_PER_PAIR;

        const r = @as(f32, @floatFromInt((combined >> THETA_BITS) & 0xF)) / R_LEVELS * max_r;
        const bucket = @as(u3, @intCast(combined & 0x7));

        const dx = r * polar_cos_table[bucket];
        const dy = r * polar_sin_table[bucket];

        sum += q[i * 2] * dx + q[i * 2 + 1] * dy;
    }

    return sum;
}

test "encode rejects odd dimension" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1.0, 2.0, 3.0 };
    const result = encode(allocator, &data, 1.0);
    try std.testing.expectError(PolarError.InvalidDimension, result);
}

test "encode rejects zero dimension" {
    const allocator = std.testing.allocator;
    const data: [0]f32 = .{};
    const result = encode(allocator, &data, 1.0);
    try std.testing.expectError(PolarError.InvalidDimension, result);
}

test "decode rejects odd dimension" {
    const allocator = std.testing.allocator;
    const compressed = [_]u8{0};
    const result = decode(allocator, &compressed, 3, 1.0);
    try std.testing.expectError(PolarError.InvalidDimension, result);
}

test "decode rejects zero dimension" {
    const allocator = std.testing.allocator;
    const compressed = [_]u8{0};
    const result = decode(allocator, &compressed, 0, 1.0);
    try std.testing.expectError(PolarError.InvalidDimension, result);
}

test "encode decode roundtrip" {
    const allocator = std.testing.allocator;
    const rotated = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const max_r: f32 = 5.0;

    const encoded = try encode(allocator, &rotated, max_r);
    defer allocator.free(encoded);

    const decoded = try decode(allocator, encoded, rotated.len, max_r);
    defer allocator.free(decoded);

    try std.testing.expectEqual(rotated.len, decoded.len);
}

test "dotProduct matches decoded dot" {
    const allocator = std.testing.allocator;
    const rotated = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const q = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const max_r: f32 = 2.0;

    const encoded = try encode(allocator, &rotated, max_r);
    defer allocator.free(encoded);

    const decoded = try decode(allocator, encoded, rotated.len, max_r);
    defer allocator.free(decoded);

    const decoded_dot = decoded[0] * q[0] + decoded[1] * q[1] + decoded[2] * q[2] + decoded[3] * q[3];
    const polar_dot = dotProduct(&q, encoded, max_r);

    try std.testing.expectEqual(decoded_dot, polar_dot);
}

test "all zero encodes safely" {
    const allocator = std.testing.allocator;
    const rotated = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    const encoded = try encode(allocator, &rotated, 1.0);
    defer allocator.free(encoded);

    const decoded = try decode(allocator, encoded, rotated.len, 1.0);
    defer allocator.free(decoded);

    for (decoded) |v| {
        try std.testing.expectEqual(0.0, v);
    }
}
