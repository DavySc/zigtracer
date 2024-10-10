const std = @import("std");

const RGB = packed struct {
    r: u8,
    g: u8,
    b: u8,
};
const Color = packed union {
    value: u24,
    rgb: RGB,
};
const PPM = struct {
    width: u32,
    height: u32,
    data: []Color,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32) !PPM {
        const self = PPM{
            .width = width,
            .height = height,
            .data = try allocator.alloc(Color, width * height),
            .allocator = allocator,
        };
        return self;
    }
    pub fn deinit(self: *PPM) void {
        self.allocator.free(self.data);
    }

    pub fn save_to_file(self: *PPM, filename: []const u8) !void {
        var file = try std.fs.cwd().createFile(filename, .{});
        defer file.close();
        errdefer file.close();

        var fwriter = file.writer();
        try fwriter.print("P3\n{} {}\n255\n", .{ self.width, self.height });

        for (self.data) |pixel| {
            try fwriter.print("{} {} {}\n", .{ pixel.rgb.r, pixel.rgb.g, pixel.rgb.b });
        }
    }
};

pub fn main() !void {
    const image_width: u32 = 256;
    const image_height: u32 = 256;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var ppm = try PPM.init(allocator, image_width, image_height);
    defer ppm.deinit();

    var j: u32 = 0;
    while (j < image_height) : (j += 1) {
        var i: u32 = 0;
        while (i < image_width) : (i += 1) {
            var r: f64 = @as(f64, @floatFromInt(i)) / @as(f64, (image_width - 1));
            var g: f64 = @as(f64, @floatFromInt(j)) / @as(f64, (image_height - 1));
            var b: f64 = 0;

            r *= 255.999;
            g *= 255.999;
            b *= 255.999;

            const index: u32 = i + j * image_width;
            const br: u8 = @intFromFloat(r);
            const bg: u8 = @intFromFloat(g);
            const bb: u8 = @intFromFloat(b);

            ppm.data[index] = Color{ .rgb = RGB{
                .r = br,
                .g = bg,
                .b = bb,
            } };
        }
    }
    try ppm.save_to_file("colorful.ppm");
}
