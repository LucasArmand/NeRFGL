// The following defines are set in main.cpp
// #define WORK_GROUP_SIZE_X

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

//layout(std430, binding=5) buffer Pos{
//    vec4 positions[];
//};
layout(std430, binding=3) buffer HiddenWeights{
    float hiddenWeights[];
};
layout(std430, binding=4) buffer InputWeights{
    float inputWeights[];
};
layout(std430, binding=5) buffer OutputWeights{
    float outputWeights[];
};
layout(std430, binding=6) buffer DFHiddenWeights{
    float df_hidden_weights[];
};
layout(std430, binding=7) buffer DFInputWeights{
    float df_input_weights[];
};
layout(std430, binding=8) buffer DFOutputWeights{
    float df_output_weights[];
};
uniform vec2 resolution;
uniform float time;

uniform sampler2D imageTexture;

vec3 sampleColor;
float sampleDensity;

const unsigned int numHiddenLayers = 4;
const unsigned int numNodesPerLayer = 40;
const unsigned int numInputs = 6;
const unsigned int numOutputs = 4;

const int MAX_ITER = 30;

float inputVec[numInputs];
float hiddenVec[numNodesPerLayer * numHiddenLayers];
float outputVec[numOutputs];

float targetVec[numOutputs];
int trainingIndex = 0;

struct raySample {
    vec3 pos;
    vec3 dir;
    vec3 color;
    float density;
    float alpha;

};

struct raySample samples[MAX_ITER];

void getSample(vec3 pos, vec3 dir) {

    inputVec[0] = pos.x;
    inputVec[1] = pos.y;
    inputVec[2] = pos.z;
    inputVec[3] = dir.x;
    inputVec[4] = dir.y;
    inputVec[5] = dir.z;
    
    for (int i = 0; i < numHiddenLayers; i++) {
        for (int j = 0; j < numNodesPerLayer; j++) {
            hiddenVec[i * numNodesPerLayer + j] = 0;
        }
    }
    for (int i = 0; i < numOutputs; i++) {
        outputVec[i] = 0;
    }

    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numNodesPerLayer; j++) {
            hiddenVec[j] += inputVec[i] * inputWeights[i * numNodesPerLayer + j];
        }
    }

    for(int i = 0; i < numNodesPerLayer; i++) {
        hiddenVec[i] = max(0, hiddenVec[i]);
        // sigmoid
        //hiddenVec[i] = 1 / (1 + exp(-hiddenVec[i]));
    }

    for (int l = 1; l < numHiddenLayers; l++) {
        for (int i = 0; i < numNodesPerLayer; i++) {
            for (int j = 0; j < numNodesPerLayer; j++) {
                hiddenVec[l * numNodesPerLayer + j] += hiddenVec[(l-1) * numNodesPerLayer + i] * hiddenWeights[(l-1) * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j];
            }
        }
        for(int i = 0; i < numNodesPerLayer; i++) {
            hiddenVec[l * numNodesPerLayer + i] = max(0, hiddenVec[l * numNodesPerLayer + i]);
            // sigmoid
            //hiddenVec[l * numNodesPerLayer + i] = 1 / (1 + exp(-hiddenVec[l * numNodesPerLayer + i]));
        }
    }
    for (int i = 0; i < numNodesPerLayer; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputVec[j] += hiddenVec[(numHiddenLayers - 1) * numNodesPerLayer + i] * outputWeights[i * numOutputs + j];
        }
    }
    for(int i = 0; i < numOutputs; i++) {
        outputVec[i] = max(0, outputVec[i]);
        //sigmoid
        //outputVec[i] = 1 / (1 + exp(-outputVec[i]));
    }

    sampleColor = vec3(outputVec[0], outputVec[1], outputVec[2]);
    sampleDensity = outputVec[3];

}



void backwards(float loss) {

    float d_input_weights[numInputs * numNodesPerLayer];
    float d_hidden_weights[numNodesPerLayer * numNodesPerLayer * (numHiddenLayers - 1)];
    float d_output_weights[numNodesPerLayer * numOutputs];

    float learning_rate = 0.05;

    float d_output_nodes[numOutputs];
    float d_hidden_nodes[numNodesPerLayer * numHiddenLayers];
    
    // dE/dwij = dE/doj * doj/dwij = dE/dojk * doj/dnetj * dnetj / dwij
    // dnetj / dwij = oi
    // https://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error

    for (int n = 0; n < MAX_ITER; n++) {
        
        struct raySample s = samples[n];
        getSample(s.pos, s.dir);

        //Last layer of weights
        for (int j = 0; j < numOutputs; j++) {
            if (outputVec[j] > 0) {
                d_output_nodes[j] = loss;
            } else {
                d_output_nodes[j] = 0;   
            }
       
        }
        for (int i = 0; i < numNodesPerLayer; i++) {
            for (int j = 0; j < numOutputs; j++) {
                d_output_weights[i * numOutputs + j] = hiddenVec[(numHiddenLayers - 1) * numNodesPerLayer + j] * d_output_nodes[j];
            }
        }
        
        //Last hidden layer
        for (int j = 0; j < numNodesPerLayer; j++) {
            //ReLU derivative
            if (hiddenVec[(numHiddenLayers - 1) * numNodesPerLayer + j] > 0) {
                for (int k = 0; k < numOutputs; k++) {
                    d_hidden_nodes[(numHiddenLayers - 1) * numNodesPerLayer + j] += d_output_nodes[k] * outputWeights[j * numOutputs + k];
                }
            }
        }
        
        //move backwards in hidden layers (destination, not including first layer)
        for (uint l = numHiddenLayers - 1; l > 0; l--) {
            for (int i = 0; i < numNodesPerLayer; i++) {
                for (int j = 0; j < numNodesPerLayer; j++) {
                    df_hidden_weights[(l-1) * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] += hiddenVec[(l-1) * numNodesPerLayer + i] * d_hidden_nodes[l * numNodesPerLayer + j];
                    if (hiddenVec[(l-1) * numNodesPerLayer + i] > 0) {
                        d_hidden_nodes[(l-1) * numNodesPerLayer + i] += hiddenWeights[(l-1) * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] * d_hidden_nodes[l * numNodesPerLayer + j];
                    }
                }
            }
        }
        
        //Input layer
        for (int i = 0; i < numInputs; i++) {
            for (int j = 0; j < numNodesPerLayer; j++) {
                df_input_weights[i * numNodesPerLayer + j] += inputVec[i] * d_hidden_nodes[j] * learning_rate;
            }
        }
        
        /*
        //Add this ray sample to the final weight delta
        for (int l = 0; l < numHiddenLayers - 1; l++) {
            for (int i = 0; i < numNodesPerLayer; i++) {
                for (int j = 0; j < numNodesPerLayer; j++) {
                    //df_hidden_weights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] += d_hidden_weights[0];// * learning_rate;
                }
            }
        }
        */
        
        for (int i = 0; i < numNodesPerLayer; i++) {
            for (int j = 0; j < numOutputs; j++) {
                df_output_weights[i * numOutputs + j] += d_output_weights[i * numOutputs + j] * learning_rate;
            }
        }
        
    }
}


void main() {

    /**/
    float MIN_RAY_DISTANCE = 1.0;
    float MAX_RAY_DISTANCE = 10.0; 
    


    float DISTANCE_THRESHOLD = 0.001;
    float min_scale = min(resolution.x, resolution.y);
    vec2 uv = vec2(gl_LocalInvocationID.x, gl_LocalInvocationID.y) / min_scale;
    float ratio = resolution.x / resolution.y;

    vec3 rayDirection = vec3(uv * 2.0 - vec2(ratio, 1.0), -1.0);
    vec3 rayOrigin = vec3(0.0, 0.0, 0.0);

   
    vec3 cameraPosition = 10 * vec3(sin(time), 0.0, cos(time));

    vec3 cameraForward_w = normalize(vec3(1.0 * sin(time), 0.0, 1.0 * cos(time)));
    vec3 cameraUp_w = normalize(vec3(0.0, 1.0, 0.0));
    vec3 cameraRight_w = cross(cameraForward_w, cameraUp_w);

    mat3 c;
    c[0] = cameraRight_w;
    c[1] = cameraUp_w;
    c[2] = cameraForward_w;

    mat3 c_i = inverse(c);

    rayOrigin = c_i * rayOrigin + cameraPosition;
    rayDirection = c_i * rayDirection;

    float t = MIN_RAY_DISTANCE;
    
    float sample_interval = (MAX_RAY_DISTANCE - MIN_RAY_DISTANCE) / MAX_ITER;
    int n = 0;
    vec3 resColor = vec3(0, 0, 0);
    float transmittance = 0.0;
    while (n < MAX_ITER) {
        vec3 pos = rayOrigin + t * rayDirection;
        getSample(pos, rayDirection);
        
        float alpha = exp(-transmittance);

        resColor += sampleColor * sampleDensity * alpha;
        t += sample_interval;
        transmittance += sampleDensity;
        samples[n].pos = pos;
        samples[n].dir = rayDirection;
        samples[n].color = sampleColor;
        samples[n].density = sampleDensity;
        samples[n].alpha = alpha;
        n++;
    }
    
    vec4 color = vec4(resColor, 1.0);
    vec4 target = texture2D(imageTexture, uv);
    vec3 diff = target.xyz - color.xyz;
    
    backwards(length(diff));
    
    //df_output_weights[0] = 5;
}