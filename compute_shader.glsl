// The following defines are set in main.cpp
// #define WORK_GROUP_SIZE_X

layout (local_size_x = WORK_GROUP_SIZE_X, local_size_y = 1, local_size_z = 1) in;

//layout(std430, binding=5) buffer Pos{
//    vec4 positions[];
//};


void main() {
    uint gid = int(gl_GlobalInvocationID.x);
    
}