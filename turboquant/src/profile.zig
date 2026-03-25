const std = @import("std");
const turboquant = @import("turboquant.zig");

const Operation = enum {
    encode,
    decode,
    dot,
    encode_prepared,
    decode_prepared,
    dot_prepared,
};

const ProfileConfig = struct {
    op: Operation,
    dim: usize,
    iterations: usize,
    seed: u32,
};

const ProfileError = error{
    MissingArgs,
    InvalidOp,
    InvalidDim,
    InvalidIterations,
    OddDimension,
    OutOfMemory,
    InvalidDimension,
};

fn parseArgs(args: [][:0]u8) ProfileError!ProfileConfig {
    if (args.len < 3) {
        return ProfileError.MissingArgs;
    }

    const op_str = args[1];
    const op: Operation = if (std.mem.eql(u8, op_str, "encode")) Operation.encode else if (std.mem.eql(u8, op_str, "decode")) Operation.decode else if (std.mem.eql(u8, op_str, "dot")) Operation.dot else if (std.mem.eql(u8, op_str, "encode_prepared")) Operation.encode_prepared else if (std.mem.eql(u8, op_str, "decode_prepared")) Operation.decode_prepared else if (std.mem.eql(u8, op_str, "dot_prepared")) Operation.dot_prepared else return ProfileError.InvalidOp;

    const dim = std.fmt.parseInt(usize, args[2], 10) catch {
        return ProfileError.InvalidDim;
    };
    if (dim == 0) return ProfileError.InvalidDim;
    if (dim % 2 != 0) return ProfileError.OddDimension;

    const iterations = if (args.len > 3)
        std.fmt.parseInt(usize, args[3], 10) catch {
            return ProfileError.InvalidIterations;
        }
    else
        1000;

    return .{
        .op = op,
        .dim = dim,
        .iterations = iterations,
        .seed = 12345,
    };
}

fn generateVector(allocator: std.mem.Allocator, dim: usize, seed: u32) ![]f32 {
    const data = try allocator.alloc(f32, dim);
    errdefer allocator.free(data);

    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (data) |*v| {
        v.* = r.float(f32) * 10 - 5;
    }

    return data;
}

fn runEncode(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !f32 {
    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    var checksum: f32 = 0;
    for (0..iterations) |_| {
        const compressed = try turboquant.encode(allocator, data, .{ .seed = seed });
        for (compressed) |b| {
            checksum += @as(f32, @floatFromInt(b));
        }
        allocator.free(compressed);
    }
    return checksum;
}

fn runDecode(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !f32 {
    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const compressed = try turboquant.encode(allocator, data, .{ .seed = seed });
    defer allocator.free(compressed);

    var checksum: f32 = 0;
    for (0..iterations) |_| {
        const decoded = try turboquant.decode(allocator, compressed, seed);
        for (decoded) |v| {
            checksum += v;
        }
        allocator.free(decoded);
    }
    return checksum;
}

fn runDot(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !f32 {
    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const query = try generateVector(allocator, dim, seed + 1);
    defer allocator.free(query);

    const compressed = try turboquant.encode(allocator, data, .{ .seed = seed });
    defer allocator.free(compressed);

    var checksum: f32 = 0;
    for (0..iterations) |_| {
        checksum += turboquant.dot(query, compressed, seed);
    }
    return checksum;
}

fn runEncodePrepared(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !f32 {
    var ptq = try turboquant.PreparedTurboQuant.prepare(allocator, dim, seed);
    defer ptq.destroy(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    var checksum: f32 = 0;
    for (0..iterations) |_| {
        const compressed = try ptq.encode(allocator, data);
        for (compressed) |b| checksum += @as(f32, @floatFromInt(b));
        allocator.free(compressed);
    }
    return checksum;
}

fn runDecodePrepared(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !f32 {
    var ptq = try turboquant.PreparedTurboQuant.prepare(allocator, dim, seed);
    defer ptq.destroy(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const compressed = try ptq.encode(allocator, data);
    defer allocator.free(compressed);

    var checksum: f32 = 0;
    for (0..iterations) |_| {
        const decoded = try ptq.decode(allocator, compressed);
        for (decoded) |v| checksum += v;
        allocator.free(decoded);
    }
    return checksum;
}

fn runDotPrepared(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !f32 {
    var ptq = try turboquant.PreparedTurboQuant.prepare(allocator, dim, seed);
    defer ptq.destroy(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const query = try generateVector(allocator, dim, seed + 1);
    defer allocator.free(query);

    const compressed = try ptq.encode(allocator, data);
    defer allocator.free(compressed);

    var checksum: f32 = 0;
    for (0..iterations) |_| {
        checksum += ptq.dot(query, compressed);
    }
    return checksum;
}

pub fn main() void {
    const args = std.process.argsAlloc(std.heap.page_allocator) catch {
        std.debug.print("error: out of memory parsing args\n", .{});
        return;
    };
    defer std.process.argsFree(std.heap.page_allocator, args);

    const config = parseArgs(args) catch |err| {
        switch (err) {
            ProfileError.MissingArgs => {
                std.debug.print("Usage: profile <op> <dim> [iterations]\n", .{});
                std.debug.print("  op: encode, decode, dot, encode_prepared, decode_prepared, dot_prepared\n", .{});
                std.debug.print("  dim: vector dimension (must be even)\n", .{});
                std.debug.print("  iterations: default 1000\n", .{});
            },
            ProfileError.InvalidOp => {
                std.debug.print("error: invalid operation '{s}'\n", .{args[1]});
            },
            ProfileError.InvalidDim => {
                std.debug.print("error: invalid dimension '{s}'\n", .{args[2]});
            },
            ProfileError.InvalidIterations => {
                std.debug.print("error: invalid iterations '{s}'\n", .{args[3]});
            },
            ProfileError.OddDimension => {
                std.debug.print("error: dimension must be even\n", .{});
            },
            else => {
                std.debug.print("error: {}\n", .{err});
            },
        }
        return;
    };

    const result: f32 = switch (config.op) {
        .encode => runEncode(std.heap.page_allocator, config.dim, config.iterations, config.seed) catch |err| {
            std.debug.print("encode error: {}\n", .{err});
            return;
        },
        .decode => runDecode(std.heap.page_allocator, config.dim, config.iterations, config.seed) catch |err| {
            std.debug.print("decode error: {}\n", .{err});
            return;
        },
        .dot => runDot(std.heap.page_allocator, config.dim, config.iterations, config.seed) catch |err| {
            std.debug.print("dot error: {}\n", .{err});
            return;
        },
        .encode_prepared => runEncodePrepared(std.heap.page_allocator, config.dim, config.iterations, config.seed) catch |err| {
            std.debug.print("encode_prepared error: {}\n", .{err});
            return;
        },
        .decode_prepared => runDecodePrepared(std.heap.page_allocator, config.dim, config.iterations, config.seed) catch |err| {
            std.debug.print("decode_prepared error: {}\n", .{err});
            return;
        },
        .dot_prepared => runDotPrepared(std.heap.page_allocator, config.dim, config.iterations, config.seed) catch |err| {
            std.debug.print("dot_prepared error: {}\n", .{err});
            return;
        },
    };

    std.debug.print("checksum: {e}\n", .{result});
}
