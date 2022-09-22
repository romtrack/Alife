// Vertex shader

struct Camera {
    view_proj: mat4x4<f32>,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}
struct InstanceInput {
    @location(5) position: vec2<f32>,
    @location(6) neighbour_count: i32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color:vec3<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(vec3<f32>(instance.position, 0.0) + model.position, 1.0);
    out.tex_coords = model.tex_coords - vec2<f32>(.5, .5);
    out.color = vec3<f32>(0.0, 1.0, 0.0);
    if (instance.neighbour_count > 15 && instance.neighbour_count <= 35 )
    {
        out.color = vec3<f32>(0.0, 0.0, 1.);
    }
    else if (instance.neighbour_count > 13 && instance.neighbour_count <= 15 )
    {
        out.color = vec3<f32>(0.58, 0.29, 0.0);
    }
    else if (instance.neighbour_count > 35)
    {
        out.color = vec3<f32>(1.0, 1.0, 0.0);
    }

    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var mask: f32;
    if (in.tex_coords.x * in.tex_coords.x + in.tex_coords.y * in.tex_coords.y < 0.5 * 0.5 )
    {
        mask = 1.0;
    } else
    {
        mask = 0.0;
    };
    return vec4<f32>(in.color, mask);
}