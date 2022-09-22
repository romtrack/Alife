use rand::distributions::{Distribution, Uniform};
use std::f32::consts::PI;

#[repr(C)]
#[derive(Copy, Clone)]
struct Point2 {
    x: f32,
    y: f32,
}

impl Point2 {
    #[inline]
    fn dist2(&self, b: &Point2) -> f32 {
        (b.x - self.x) * (b.x - self.x) + (b.y - self.y) * (b.y - self.y)
    }
}

unsafe impl bytemuck::Zeroable for Point2 {
    fn zeroed() -> Self {
        Point2 { x: 0., y: 0. }
    }
}

unsafe impl bytemuck::Pod for Point2 {}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    position: Point2, // Waiting for cgmath to have bytemuck.
    heading: f32,
    neighbour_count: i32,
}

impl Particle {
    fn new() -> Self {
        Particle {
            position: Point2 { x: 0., y: 0. },
            heading: 0.,
            neighbour_count: 0,
        }
    }

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Particle>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress, // [f32; 3] Point2D + f32 for heading.
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
                    shader_location: 6,
                    format: wgpu::VertexFormat::Sint32,
                },
            ],
        }
    }
}

#[derive(Copy, Clone)]
pub struct Params {
    pub alpha: f32, // Fixed Rotation.
    pub beta: f32,  // Rotation proportional to neighbourhood.
    pub r: f32,     // Interaction radius.
    pub v: f32,     // Speed.
}

pub struct Pps {
    population: Vec<Particle>,
    population_back: Vec<Particle>,
    system_params: Params,
    domain: f32,
}

impl Pps {
    pub fn new(domain_size: f32, population_count: usize, system_params: Params) -> Self {
        let mut population = Vec::with_capacity(population_count);
        let size = domain_size * 0.5;
        let position_distrib = Uniform::from(-size..size);
        let heading_distrib = Uniform::from(-PI..PI);
        let mut rng = rand::thread_rng();

        for _ in 0..population_count {
            population.push({
                Particle {
                    position: Point2 {
                        x: position_distrib.sample(&mut rng),
                        y: position_distrib.sample(&mut rng),
                    },
                    heading: heading_distrib.sample(&mut rng),
                    neighbour_count: 0,
                }
            });
        }

        Self {
            population,
            population_back: vec![Particle::new(); population_count],
            system_params,
            domain: size,
        }
    }

    pub fn step(&mut self) {
        fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        }
        let mut cell_list = cell_list::CellList::new(self.system_params.r);
        cell_list.build(&self.population);

        for i in 0..self.population.len() {
            let mut current_particle = self.population[i];
            let mut l: i32 = 0;
            let mut r: i32 = 0;
            let right_heading = current_particle.heading + std::f32::consts::FRAC_PI_2;
            let r_normal_vector = [right_heading.cos(), right_heading.sin(), 0.];

            for j in cell_list.get_neighbours(&current_particle) {
                //for j in 0..self.population.len() {
                let other_particle = self.population[j];
                if current_particle.position.dist2(&other_particle.position)
                    < self.system_params.r * self.system_params.r
                {
                    let i_to_j = [
                        other_particle.position.x - current_particle.position.x,
                        other_particle.position.y - current_particle.position.y,
                        0.,
                    ];
                    if dot(&r_normal_vector, &i_to_j) < 0. {
                        // Left.
                        l += 1;
                    } else {
                        // Right.
                        r += 1;
                    }
                }
            }
            // -self.system_params.alpha because positive alpha means turning right.
            current_particle.heading += -self.system_params.alpha
                + self.system_params.beta * (l + r) as f32 * (r - l).signum() as f32;
            current_particle.position.x += current_particle.heading.cos() * self.system_params.v;
            current_particle.position.y += current_particle.heading.sin() * self.system_params.v;
            current_particle.position.x =
                current_particle.position.x.clamp(-self.domain, self.domain);
            current_particle.position.y =
                current_particle.position.y.clamp(-self.domain, self.domain);
            current_particle.neighbour_count = l + r;
            self.population_back[i] = current_particle;
        }

        std::mem::swap(&mut self.population, &mut self.population_back);
    }

    pub fn particles(&self) -> &[Particle] {
        &self.population
    }
}
pub mod cell_list {
    use crate::pps::Particle;
    use std::collections::HashMap;

    type Index = (i32, i32);

    pub struct CellList {
        cells: HashMap<Index, Vec<usize>>,
        radius: f32,
    }

    impl CellList {
        pub fn new(radius: f32) -> Self {
            Self {
                radius,
                cells: Default::default(),
            }
        }

        pub fn build(&mut self, particles: &[Particle]) {
            for (i, p) in particles.iter().enumerate() {
                let index = (
                    (p.position.x / self.radius) as i32,
                    (p.position.y / self.radius) as i32,
                );
                self.cells.entry(index).or_default().push(i);
            }
        }

        pub fn get_neighbours(&self, particle: &Particle) -> Vec<usize> {
            let mut res = Vec::with_capacity(100);
            let index = (
                (particle.position.x / self.radius) as i32,
                (particle.position.y / self.radius) as i32,
            );

            for x in -1..=1 {
                for y in -1..=1 {
                    let cell_index = (index.0 + x, index.1 + y);
                    if let Some(v) = self.cells.get(&cell_index) {
                        res.extend_from_slice(v);
                    }
                }
            }

            res
        }
    }
}
