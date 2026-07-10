use crate::SDFField;
use bevy::{
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSystems,
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
    },
};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompactionUniforms {
    pub chunk_size: u32,
    pub total_cells: u32,
    pub iso_level: f32,
    pub _pad0: u32,
}

#[derive(Component)]
pub struct GpuVoxelChunkBuffers {
    pub chunk_coord: IVec3,
    pub sdf_view: TextureView,
    pub jfa_view: TextureView,

    // --- Vertex Generation/Compaction Buffers ---
    pub flags_buffer: Buffer,             // Pass 1 Output (Storage)
    pub compacted_offsets_buffer: Buffer, // Pass 2 Output (Storage)
    pub scattered_vertex_buffer: Buffer,  // Pass 3 Temporary (Storage)
    pub final_vertex_buffer: Buffer,      // Pass 3 Output Packed (Storage | Vertex)

    pub index_buffer: Buffer,         // Pass 4 Output (Storage | Index)
    pub indirect_args_buffer: Buffer, // Pass 4 Counter (Storage | Indirect)
    pub bind_group: BindGroup,
}

#[derive(Resource)]
pub struct VoxelPipelineLayouts {
    pub chunk_bind_group_layout: BindGroupLayout,
}

pub struct VoxelRenderPlugin;

impl Plugin for VoxelRenderPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        let render_device = render_app.world().resource::<RenderDevice>();

        let bind_group_layout = render_device.create_bind_group_layout(
            "voxel_chunk_layout",
            &[
                // Binding 0: SDF Volume (ReadOnly Storage)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                },
                // Binding 1: Flags Buffer (Pass 1 tracking)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Compacted Offsets / Prefix Sum Array
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Scattered Uncompacted Vertex Buffer
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Final Compacted Vertex Buffer
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE | ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Indirect Arguments Buffer (Atomic Appended)
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        render_app.insert_resource(VoxelPipelineLayouts {
            chunk_bind_group_layout: bind_group_layout,
        });
    }
}
