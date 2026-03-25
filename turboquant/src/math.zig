const std = @import("std");

pub fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0;
    for (a, b) |av, bv| {
        sum += av * bv;
    }
    return sum;
}

pub fn norm(x: []const f32) f32 {
    return @sqrt(dot(x, x));
}

pub fn scale(v: []f32, s: f32) void {
    for (v) |*val| {
        val.* *= s;
    }
}

pub fn addScaled(out: []f32, a: []const f32, b: []const f32, scale_b: f32) void {
    std.debug.assert(out.len == a.len);
    std.debug.assert(a.len == b.len);
    for (0..a.len) |i| {
        out[i] = a[i] + b[i] * scale_b;
    }
}

pub fn sub(a: []const f32, b: []const f32, out: []f32) void {
    std.debug.assert(out.len == a.len);
    std.debug.assert(a.len == b.len);
    for (0..a.len) |i| {
        out[i] = a[i] - b[i];
    }
}

pub fn copy(src: []const f32, dst: []f32) void {
    std.debug.assert(dst.len == src.len);
    @memcpy(dst, src);
}

pub fn zero(v: []f32) void {
    @memset(v, 0);
}

test "dot simple" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };
    try std.testing.expectEqual(32.0, dot(&a, &b));
}

test "norm known vector" {
    const v = [_]f32{ 3.0, 4.0 };
    try std.testing.expectEqual(5.0, norm(&v));
}

test "scale in place" {
    var v = [_]f32{ 1.0, 2.0, 3.0 };
    scale(&v, 2.0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 2.0, 4.0, 6.0 }, &v);
}

test "sub produces residual" {
    const a = [_]f32{ 5.0, 10.0 };
    const b = [_]f32{ 2.0, 3.0 };
    var out: [2]f32 = undefined;
    sub(&a, &b, &out);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3.0, 7.0 }, &out);
}

test "copy exact" {
    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var dst: [4]f32 = undefined;
    copy(&src, &dst);
    try std.testing.expectEqualSlices(f32, &src, &dst);
}

test "zero clears" {
    var v = [_]f32{ 1.0, 2.0, 3.0 };
    zero(&v);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 0.0, 0.0, 0.0 }, &v);
}
