#version 430 core
precision highp float;
uniform vec2 resolution;
uniform float time;

out vec4 fragColor;

layout(std430, binding=3) buffer HiddenWeights{
    float hiddenWeights[];
};
layout(std430, binding=4) buffer InputWeights{
    float inputWeights[];
};
layout(std430, binding=5) buffer OutputWeights{
    float outputWeights[];
};


vec3 sampleColor;
float sampleDensity;

const unsigned int numHiddenLayers = 3;
const unsigned int numNodesPerLayer = 10;
const unsigned int numInputs = 6;
const unsigned int numOutputs = 4;

float inputVec[numInputs];
float hiddenVec[numNodesPerLayer * numHiddenLayers];
float outputVec[numOutputs];

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
        hiddenVec[i] = 1 / (1 + exp(-hiddenVec[i]));
    }

    for (int l = 1; l < numHiddenLayers; l++) {
        for (int i = 0; i < numNodesPerLayer; i++) {
            for (int j = 0; j < numNodesPerLayer; j++) {
                hiddenVec[l * numNodesPerLayer + j] += hiddenVec[(l-1) * numNodesPerLayer + i] * hiddenWeights[(l-1) * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j];
            }
        }
        for(int i = 0; i < numNodesPerLayer; i++) {
            hiddenVec[l * numNodesPerLayer + i] = 1 / (1 + exp(-hiddenVec[l * numNodesPerLayer + i]));
        }
    }
    for (int i = 0; i < numNodesPerLayer; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputVec[j] += hiddenVec[(numHiddenLayers - 1) * numNodesPerLayer + i] * outputWeights[i * numNodesPerLayer + j];
        }
    }
    for(int i = 0; i < numOutputs; i++) {
        outputVec[i] = 1 / (1 + exp(-outputVec[i]));
    }

    sampleColor = vec3(outputVec[0], outputVec[1], outputVec[2]);
    sampleDensity = outputVec[3];


/*
    sampleColor = pos;
    sampleDensity = exp(-length(pos)) * sin(pos.x + time) * cos(pos.z + time * 0.7);
    if (pos.x < 1.0 && pos.x > 0.0 && pos.y < 1.0 && pos.y > 0.0 && pos.z < 1.0 && pos.z > 0) {
        sampleDensity = 1.0;
        sampleColor = vec3(1.0, 1.0, 1.0);
    }
    if (pos.x < 1.0 && pos.x > 0.0 && pos.y < 1.0 && pos.y > 0.0 && pos.z < 2.0 && pos.z > 1.0) {
        sampleDensity = 2.0;
        sampleColor = vec3(0.0, 1.0, 0.0);
    }
*/
}




void main()
{
    vec3 pointLight = vec3(5.0 * cos(time), 0.0, 5.0 * sin(time));

    float MIN_RAY_DISTANCE = 0.01;
    float MAX_RAY_DISTANCE = 1000.0;
    int MAX_ITER = 15;
    
    float DISTANCE_THRESHOLD = 0.001;
    float min_scale = min(resolution.x, resolution.y);
    vec2 uv = gl_FragCoord.xy / min_scale;
    float ratio = resolution.x / resolution.y;

    vec3 rayDirection = vec3(uv * 2.0 - vec2(ratio, 1.0), -1.0);
    vec3 rayOrigin = vec3(0.0, 0.0, 3.0);
    float t = MIN_RAY_DISTANCE;
    
    float sample_interval = (MAX_RAY_DISTANCE - MIN_RAY_DISTANCE) / MAX_ITER;
    int n = 0;
    vec3 color = vec3(0, 0, 0);
    float transmittance = 0.0;
    while (n < MAX_ITER) {
        vec3 pos = rayOrigin + t * rayDirection;
        getSample(pos, rayDirection);
        
        color += sampleColor * sampleDensity * exp(-transmittance);
        t += sample_interval;
        transmittance += sampleDensity;
        n++;
    }

    fragColor = vec4(color, 1.0);
    
}