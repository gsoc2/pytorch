import glob
import os
import tempfile
import unittest

from tools.gen_vulkan_spv import DEFAULT_ENV, SPVGenerator
from yaml.constructor import ConstructorError

######################
## Data for testing ##
######################

test_shader = """
#version 450 core

#define FORMAT ${FORMAT}
#define PRECISION ${PRECISION}

#define OP(X) ${OPERATOR}

$def is_int(dtype):
$   return dtype in {"int", "int32", "int8"}

$def is_uint(dtype):
$   return dtype in {"uint", "uint32", "uint8"}

$if is_int(DTYPE):
  #define VEC4_T ivec4
$elif is_uint(DTYPE):
  #define VEC4_T uvec4
$else:
  #define VEC4_T vec4

$if not INPLACE:
  $if is_int(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly iimage3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION isampler3D uInput;
  $elif is_uint(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly uimage3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION usampler3D uInput;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
$else:
  $if is_int(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict iimage3D uOutput;
  $elif is_uint(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict uimage3D uOutput;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  $if not INPLACE:
    VEC4_T v = texelFetch(uInput, pos, 0);
  $else:
    VEC4_T v = imageLoad(uOutput, pos);

  $for i in range(ITER[0]):
    for (int i = 0; i < ${ITER[1]}; ++i) {
        v = OP(v + i);
    }

  imageStore(uOutput, pos, OP(v));
}

"""

test_shader_1_inplace_float_expected = """
#version 450 core

#define FORMAT rgba16f
#define PRECISION highp

#define OP(X) X + 3



#define VEC4_T vec4

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  VEC4_T v = imageLoad(uOutput, pos);

  for (int i = 0; i < 5; ++i) {
      v = OP(v + i);
  }
  for (int i = 0; i < 5; ++i) {
      v = OP(v + i);
  }
  for (int i = 0; i < 5; ++i) {
      v = OP(v + i);
  }

  imageStore(uOutput, pos, OP(v));
}
"""

test_shader_2_uint8_expected = """
#version 450 core

#define FORMAT rgba8ui
#define PRECISION highp

#define OP(X) X * 2



#define VEC4_T uvec4

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly uimage3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION usampler3D uInput;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  VEC4_T v = texelFetch(uInput, pos, 0);

  for (int i = 0; i < 4; ++i) {
      v = OP(v + i);
  }
  for (int i = 0; i < 4; ++i) {
      v = OP(v + i);
  }

  imageStore(uOutput, pos, OP(v));
}
"""

test_shader_3_int_expected = """
#version 450 core

#define FORMAT rgba32i
#define PRECISION highp

#define OP(X) X - 1



#define VEC4_T ivec4

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION isampler3D uInput;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  VEC4_T v = texelFetch(uInput, pos, 0);

  for (int i = 0; i < 2; ++i) {
      v = OP(v + i);
  }
  for (int i = 0; i < 2; ++i) {
      v = OP(v + i);
  }
  for (int i = 0; i < 2; ++i) {
      v = OP(v + i);
  }

  imageStore(uOutput, pos, OP(v));
}
"""

test_params_yaml = """
test_shader:
  parameter_names_with_default_values:
    DTYPE: float
    INPLACE: false
    OPERATOR: X + 3
    ITER: !!python/tuple [3, 5]
  generate_variant_forall:
    INPLACE:
      - VALUE: false
        SUFFIX: ""
      - VALUE: true
        SUFFIX: inplace
    DTYPE:
      - VALUE: int8
      - VALUE: uint8
      - VALUE: int
      - VALUE: uint
      - VALUE: float
  shader_variants:
    - NAME: test_shader_1
    - NAME: test_shader_2
      OPERATOR: X * 2
      ITER: !!python/tuple [2, 4]
    - NAME: test_shader_3
      OPERATOR: X - 1
      ITER: !!python/tuple [3, 2]
      generate_variant_forall:
        DTYPE:
        - VALUE: float
        - VALUE: int

"""

################
## Unit Tests ##
################


class TestVulkanSPVCodegen(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

        with open(f"{self.tmpdir.name}/test_shader.glsl,", "w") as f:
            f.write(test_shader)

        with open(f"{self.tmpdir.name}/test_params.yaml", "w") as f:
            f.write(test_params_yaml)

        self.tmpoutdir = tempfile.TemporaryDirectory()

        self.generator = SPVGenerator(
            src_dir_paths=self.tmpdir.name, env=DEFAULT_ENV, glslc_path=None
        )

    def cleanUp(self):
        self.tmpdir.cleanup()
        self.tmpoutdir.cleanup()

    def testOutputMap(self):
        # Each shader variant will produce variants generated based on all possible combinations
        # of the DTYPE and INPLACE parameters. test_shader_3 has fewer generated variants due to
        # a custom specified generate_variant_forall field.
        expected_output_shaders = {
            "test_shader_1_float",
            "test_shader_1_inplace_float",
            "test_shader_1_inplace_int",
            "test_shader_1_inplace_int8",
            "test_shader_1_inplace_uint",
            "test_shader_1_inplace_uint8",
            "test_shader_1_int",
            "test_shader_1_int8",
            "test_shader_1_uint",
            "test_shader_1_uint8",
            "test_shader_2_float",
            "test_shader_2_inplace_float",
            "test_shader_2_inplace_int",
            "test_shader_2_inplace_int8",
            "test_shader_2_inplace_uint",
            "test_shader_2_inplace_uint8",
            "test_shader_2_int",
            "test_shader_2_int8",
            "test_shader_2_uint",
            "test_shader_2_uint8",
            "test_shader_3_float",
            "test_shader_3_int",
        }

        actual_output_shaders = set(self.generator.output_shader_map.keys())

        self.assertEqual(expected_output_shaders, actual_output_shaders)

    def testGeneratedGLSL(self):
        self.generator.generateSPV(self.tmpoutdir.name)

        ## Check some of the generated GLSL files against golden data

        with open(f"{self.tmpoutdir.name}/test_shader_1_inplace_float.glsl", "rb") as f:
            contents = str(f.read().strip(), "utf-8")
            self.assertEqual(contents, test_shader_1_inplace_float_expected.strip())

        with open(f"{self.tmpoutdir.name}/test_shader_2_uint8.glsl", "rb") as f:
            contents = str(f.read().strip(), "utf-8")
            self.assertEqual(contents, test_shader_2_uint8_expected.strip())

        with open(f"{self.tmpoutdir.name}/test_shader_3_int.glsl", "rb") as f:
            contents = str(f.read().strip(), "utf-8")
            self.assertEqual(contents, test_shader_3_int_expected.strip())

        ## For the rest, perform basic checks

        def check_file(filepath):
            with open(filepath, "rb") as f:
                contents = str(f.read(), "utf-8")

                if "inplace" in filepath:
                    self.assertTrue("VEC4_T v = imageLoad(uOutput, pos);" in contents)
                else:
                    self.assertTrue(
                        "VEC4_T v = texelFetch(uInput, pos, 0);" in contents
                    )

                if "_float" in filepath:
                    self.assertTrue("#define FORMAT rgba16f" in contents)
                elif "_int8" in filepath:
                    self.assertTrue("#define FORMAT rgba8i" in contents)
                elif "_uint8" in filepath:
                    self.assertTrue("#define FORMAT rgba8ui" in contents)
                elif "_int" in filepath and "int8" not in filepath:
                    self.assertTrue("#define FORMAT rgba32i" in contents)
                elif "_uint" in filepath and "uint8" not in filepath:
                    self.assertTrue("#define FORMAT rgba32ui" in contents)

        file_list = glob.glob(f"{self.tmpoutdir.name}/**/*.glsl", recursive=True)
        for file in file_list:
            check_file(file)
