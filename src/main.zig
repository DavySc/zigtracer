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

        const fwriter = file.writer();
        var bufferedWriter = std.io.bufferedWriter(fwriter);
        var bwriter = bufferedWriter.writer();
        try bwriter.print("P3\n{} {}\n255\n", .{ self.width, self.height });

        for (self.data) |pixel| {
            try bwriter.print("{} {} {}\n", .{ pixel.rgb.r, pixel.rgb.g, pixel.rgb.b });
        }
        try bufferedWriter.flush();
    }
};

const Vec3 = struct {
    data: @Vector(3, f64),

    pub fn init(xd: f64, yd: f64, zd: f64) Vec3 {
        var self = Vec3{ .data = @Vector(3, f64){ 0.0, 0.0, 0.0 } };
        self.data[0] = xd;
        self.data[1] = yd;
        self.data[2] = zd;

        return self;
    }

    pub fn zero() Vec3 {
        return Vec3{ .data = @Vector(3, f64){ 0.0, 0.0, 0.0 } };
    }
    pub fn x(self: *const Vec3) f64 {
        return self.data[0];
    }
    pub fn y(self: *const Vec3) f64 {
        return self.data[1];
    }
    pub fn z(self: *const Vec3) f64 {
        return self.data[2];
    }

    pub fn negate(self: *const Vec3) *Vec3 {
        self.data = -self.data;
        return self;
    }

    pub fn addEq(self: *const Vec3, v: Vec3) *Vec3 {
        self.data += v.data;
        return self;
    }

    pub fn mulEq(self: *const Vec3, scalar: f64) *Vec3 {
        self.data *= @splat(scalar);
        return self;
    }

    pub fn divEq(self: *const Vec3, scalar: f64) *Vec3 {
        self.data /= @splat(scalar);
        return self;
    }

    pub fn length_squared(self: *const Vec3) f64 {
        return self.data[0] * self.data[0] + self.data[1] * self.data[1] + self.data[2] * self.data[2];
    }

    pub fn length(self: *const Vec3) f64 {
        return std.math.sqrt(self.length_squared());
    }

    pub fn add(u: *const Vec3, v: Vec3) Vec3 {
        return Vec3{ .data = u.data + v.data };
    }

    pub fn sub(u: *const Vec3, v: Vec3) Vec3 {
        return Vec3{ .data = u.data - v.data };
    }

    pub fn mul(u: *const Vec3, v: Vec3) Vec3 {
        return Vec3{ .data = u.data * v.data };
    }

    pub fn mul_scalar(u: *const Vec3, v: f64) Vec3 {
        return Vec3{ .data = u.data * @as(@Vector(3, f64), @splat(v)) };
    }

    pub fn div(u: *const Vec3, v: f64) Vec3 {
        return Vec3{ .data = u.data / @as(@Vector(3, f64), @splat(v)) };
    }

    pub fn dot(u: *const Vec3, v: Vec3) f64 {
        const result: f64 = @reduce(.Add, u.data * v.data);
        return result;
    }

    pub fn cross(u: *const Vec3, v: Vec3) Vec3 {
        const xd: f64 = u.data[1] * v.data[2] - u.data[2] * v.data[1];
        const yd: f64 = u.data[2] * v.data[0] - u.data[0] * v.data[2];
        const zd: f64 = u.data[1] * v.data[0] - u.data[1] * v.data[0];
        return init(xd, yd, zd);
    }

    pub fn unit_vector(v: *const Vec3) Vec3 {
        return v.div(v.length());
    }
};

const Point3 = Vec3;
const Color3 = Vec3;

pub fn color3_to_color(c: Color3) Color {
    return Color{ .rgb = RGB{
        .r = @intFromFloat(c.x() * 255.99),
        .g = @intFromFloat(c.y() * 255.99),
        .b = @intFromFloat(c.z() * 255.99),
    } };
}

const Ray = struct {
    origin: Vec3,
    direction: Vec3,

    pub fn init(origin: Vec3, direction: Vec3) Ray {
        return Ray{
            .origin = origin,
            .direction = direction,
        };
    }
    pub fn at(self: *const Ray, t: f64) Point3 {
        return self.origin.add(self.direction.mul_scalar(t));
    }
};

const HitRecord = struct {
    point: Point3,
    normal: Vec3,
    t: f64,
    front_face: bool,

    fn set_face_normal(self: *HitRecord, ray: Ray, outward_normal: Vec3) void {
        self.front_face = ray.direction.dot(outward_normal) < 0;
        self.normal = if (self.front_face) outward_normal else outward_normal.negate();
    }
};

const Hittable = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        hit: *const fn (ctx: *anyopaque, ray: Ray, ray_tmin: f64, ray_tmax: f64, hit_record: *HitRecord, ret_addr: usize) bool,
    };

    pub fn rawHit(self: Hittable, ray: Ray, ray_tmin: f64, ray_tmax: f64, hit_record: *HitRecord, ret_addr: usize) bool {
        return self.vtable.hit(self.ptr, ray, ray_tmin, ray_tmax, hit_record, ret_addr);
    }
};

const Sphere = struct {
    center: Point3,
    radius: f64,
    pub fn init(center: Point3, radius: f64) Sphere {
        return Sphere{ .center = center, .radius = radius };
    }

    pub fn hit(ctx: *anyopaque, ray: Ray, ray_tmin: f64, ray_tmax: f64, hit_record: *HitRecord, ret_addr: usize) bool {
        _ = ret_addr;
        const self: *Sphere = @ptrCast(@alignCast(ctx));
        const oc = ray.origin.sub(self.center);
        const a = ray.direction.length_squared();
        const h = oc.dot(ray.direction);
        const c = oc.length_squared() - self.radius * self.radius;

        const discriminant = h * h - a * c;

        if (discriminant < 0) {
            return false;
        }
        const sqrtd = std.math.sqrt(discriminant);

        var root = (h - sqrtd) / a;
        if (root <= ray_tmin or ray_tmax <= root) {
            root = (h + sqrtd) / a;
            if (root <= ray_tmin or ray_tmax <= root) {
                return false;
            }
        }
        hit_record.t = root;
        hit_record.point = ray.at(root);
        hit_record.normal = (hit_record.point.sub(self.center).divEq(self.radius));
        hit_record.outward_normal = hit_record.point.sub(self.center).divEq(self.radius);
        hit_record.set_face_normal(ray, hit_record.outward_normal);
        return true;
    }

    pub fn hittable(self: *Sphere) Hittable {
        return .{
            .ptr = self,
            .vtable = &.{
                .hit = hit,
            },
        };
    }
};

pub fn hit_sphere(center: Point3, radius: f64, ray: Ray) f64 {
    const oc = center.sub(ray.origin);
    const a = ray.direction.length_squared();
    const h = ray.direction.dot(oc);
    const c = oc.length_squared() - radius * radius;
    const discriminant = h * h - a * c;

    return if (discriminant < 0)
        -1.0
    else
        (h - std.math.sqrt(discriminant)) / a;
}

pub fn ray_color(ray: Ray) Color3 {
    const t = hit_sphere(Point3.init(0, 0, -1), 0.5, ray);
    if (t > 0.0) {
        const normal_vector = ray.at(t).sub(Point3.init(0, 0, -1)).unit_vector();
        return Color3.init(normal_vector.x() + 1.0, normal_vector.y() + 1.0, normal_vector.z() + 1.0).mul_scalar(0.5);
    }
    const unit_direction: Vec3 = ray.direction.unit_vector();
    const a: f64 = 0.5 * (unit_direction.y() + 1.0);
    return Color3.init(1.0, 1.0, 1.0).mul_scalar(1.0 - a).add(Color3.init(0.5, 0.7, 1.0).mul_scalar(a));
}

pub fn main() !void {
    // Image
    const aspect_ratio = 16.0 / 9.0;
    const image_width: u32 = 512;
    const image_height: u32 = @intFromFloat(@as(f64, @floatFromInt(image_width)) / aspect_ratio);

    // camera
    const focal_length: f64 = 1.0;
    const viewport_height: f64 = 2.0;
    const viewport_width: f64 = viewport_height * @as(f64, @floatFromInt(image_width)) / @as(f64, @floatFromInt(image_height));
    var camera_center = Point3.init(0, 0, 0);

    var viewport_u = Vec3.init(viewport_width, 0, 0);
    var viewport_v = Vec3.init(0, -viewport_height, 0);

    var pixel_delta_u = viewport_u.div(@floatFromInt(image_width));
    var pixel_delta_v = viewport_v.div(@floatFromInt(image_height));
    var viewport_upper_left = camera_center.sub(Vec3.init(0, 0, focal_length)).sub(viewport_u.div(2)).sub(viewport_v.div(2));
    var pixel00_loc = viewport_upper_left.add(pixel_delta_u.add(pixel_delta_v.mul_scalar(0.5)));

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var ppm = try PPM.init(allocator, image_width, image_height);
    defer ppm.deinit();

    var j: u32 = 0;
    while (j < image_height) : (j += 1) {
        std.log.info("\rScanlines remaining: {}", .{image_height - j});
        var i: u32 = 0;
        while (i < image_width) : (i += 1) {
            var pixel_center = pixel00_loc.add(pixel_delta_u.mul_scalar(@floatFromInt(i))).add(pixel_delta_v.mul_scalar(@floatFromInt(j)));
            const ray_diretion = pixel_center.sub(camera_center);
            const ray = Ray.init(camera_center, ray_diretion);
            const col = ray_color(ray);

            const index: u32 = i + j * image_width;
            ppm.data[index] = color3_to_color(col);
        }
    }
    try ppm.save_to_file("colorful.ppm");
}
