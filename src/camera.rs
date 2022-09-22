use crate::InputEvent;
use std::f32;

use winit::event::*;

#[rustfmt::skip]
const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub struct Camera {
    pub eye: cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        // 1.
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        // 2.
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        // 3.
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

pub struct CameraController {
    speed: f32,
    translation_enabled: bool,
    zoom_enabled: bool,
    delta: (f32, f32),
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            translation_enabled: false,
            zoom_enabled: false,
            delta: (0., 0.),
        }
    }

    pub fn process_events(&mut self, event: InputEvent) -> bool {
        match event {
            InputEvent::WindowEvent(event) => match event {
                WindowEvent::MouseInput { button, state, .. } => match button {
                    MouseButton::Middle => {
                        if *state == ElementState::Pressed {
                            self.translation_enabled = true;
                        } else {
                            self.translation_enabled = false;
                        }
                        true
                    }
                    MouseButton::Right => {
                        if *state == ElementState::Pressed {
                            self.zoom_enabled = true;
                        } else {
                            self.zoom_enabled = false;
                        }
                        true
                    }
                    _ => false,
                },
                _ => false,
            },
            InputEvent::DeviceEvent(event) => match event {
                DeviceEvent::MouseMotion { delta } => {
                    self.delta.0 = delta.0 as f32;
                    self.delta.1 = delta.1 as f32;
                    true
                }
                _ => false,
            },
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        use cgmath::InnerSpace;

        let forward = (camera.target - camera.eye).normalize();
        let right = forward.cross(camera.up).normalize();
        let up = forward.cross(right).normalize();

        if self.translation_enabled {
            let mut displacement = right * self.speed * self.delta.0;
            displacement += up * self.speed * self.delta.1;
            camera.eye -= displacement;
            camera.target -= displacement;
        } else if self.zoom_enabled {
            let displacement = forward
                * self.speed
                * -self.delta.0.signum()
                * (self.delta.0 * self.delta.0 + self.delta.1 * self.delta.1).sqrt();
            camera.eye -= displacement;
            camera.target -= displacement;
        }

        self.delta = (0., 0.);
    }
}
