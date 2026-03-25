const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const turboquant_mod = b.addModule("turboquant", .{
        .root_source_file = b.path("src/turboquant.zig"),
        .target = target,
    });
    turboquant_mod.addImport("matrix", b.addModule("matrix", .{
        .root_source_file = b.path("src/matrix.zig"),
        .target = target,
    }));
    turboquant_mod.addImport("polar", b.addModule("polar", .{
        .root_source_file = b.path("src/polar.zig"),
        .target = target,
    }));
    turboquant_mod.addImport("qjl", b.addModule("qjl", .{
        .root_source_file = b.path("src/qjl.zig"),
        .target = target,
    }));
    turboquant_mod.addImport("format", b.addModule("format", .{
        .root_source_file = b.path("src/format.zig"),
        .target = target,
    }));
    turboquant_mod.addImport("rotation", b.addModule("rotation", .{
        .root_source_file = b.path("src/rotation.zig"),
        .target = target,
    }));
    turboquant_mod.addImport("math", b.addModule("math", .{
        .root_source_file = b.path("src/math.zig"),
        .target = target,
    }));

    const deps = b.dependency("zbench", .{
        .target = target,
        .optimize = optimize,
    });
    const zbench_mod = deps.module("zbench");

    const tests = b.addTest(.{
        .root_module = turboquant_mod,
    });

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);

    const bench_mod = b.addModule("bench_main", .{
        .root_source_file = b.path("bench/main.zig"),
    });
    bench_mod.addImport("turboquant", turboquant_mod);
    bench_mod.addImport("zbench", zbench_mod);

    const bench = b.addTest(.{
        .root_module = bench_mod,
    });

    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&b.addRunArtifact(bench).step);
}
