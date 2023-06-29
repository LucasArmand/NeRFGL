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
layout(std430, binding=6) buffer HiddenVec{
    float hiddenVec[];
};
layout(std430, binding=7) buffer InputNodes{
    float inputVec[];
};
layout(std430, binding=8) buffer OutputVec{
    float outputVec[];
};
layout(std430, binding=10) buffer HiddenBiasWeights{
    float hiddenBiasWeights[];
};
layout(std430, binding=11) buffer OutputBiasWeights{
    float outputBiasWeights[];
};
layout(std430, binding=9) buffer Loss{
    float loss;
};
uniform vec2 resolution;
uniform float time;

uniform sampler2D imageTexture;

vec3 sampleColor;
float sampleDensity;



const unsigned int numHiddenLayers = 5;
const unsigned int numNodesPerLayer = 5;
const unsigned int numInputs = 3;
const unsigned int numOutputs = 1;

const int MAX_ITER = 5;

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

float forward(vec3 pos) {

    inputVec[0] = pos.x;
    inputVec[1] = pos.y;
    inputVec[2] = pos.z;

    
    for (int i = 0; i < numHiddenLayers; i++) {
        for (int j = 0; j < numNodesPerLayer; j++) {
            hiddenVec[i * numNodesPerLayer + j] = 0.0;
        }
    }

    for (int i = 0; i < numOutputs; i++) {
        outputVec[i] = 0.0;
    }

    for (int j = 0; j < numNodesPerLayer; j++) {
        for (int i = 0; i < numInputs; i++) {
            hiddenVec[j] += inputVec[i] * inputWeights[i * numNodesPerLayer + j];
        }
        hiddenVec[j] += hiddenBiasWeights[j];
     }
    

    for(int i = 0; i < numNodesPerLayer; i++) {
        hiddenVec[i] = max(0, hiddenVec[i]);
        // sigmoid
        //hiddenVec[i] = 1 / (1 + exp(-hiddenVec[i]));
    }

    for (int l = 1; l < numHiddenLayers; l++) {
        for (int j = 0; j < numNodesPerLayer; j++) {
            for (int i = 0; i < numNodesPerLayer; i++) {
                hiddenVec[l * numNodesPerLayer + j] += hiddenVec[(l-1) * numNodesPerLayer + i] * hiddenWeights[(l-1) * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j];
            }
            hiddenVec[l * numNodesPerLayer + j] += hiddenBiasWeights[l * numNodesPerLayer + j];
        }
        for(int i = 0; i < numNodesPerLayer; i++) {
            hiddenVec[l * numNodesPerLayer + i] = max(0, hiddenVec[l * numNodesPerLayer + i]);
            // sigmoid
            //hiddenVec[l * numNodesPerLayer + i] = 1 / (1 + exp(-hiddenVec[l * numNodesPerLayer + i]));
        }
    }
    for (int j = 0; j < numOutputs; j++) {
        for (int i = 0; i < numNodesPerLayer; i++) {
            outputVec[j] += hiddenVec[(numHiddenLayers - 1) * numNodesPerLayer + i] * outputWeights[i * numOutputs + j];
        }
        outputVec[j] += outputBiasWeights[j];
    }
    for(int i = 0; i < numOutputs; i++) {
        //outputVec[i] = max(0, outputVec[i]);
        //sigmoid
        //outputVec[i] = 1 / (1 + exp(-outputVec[i]));
    }
    
    return outputVec[0];

    /*
    sampleColor = vec3(outputVec[0], outputVec[1], outputVec[2]);
    sampleDensity = outputVec[3];
    */

}



void backwards(float diff) {

    float d_input_weights[numInputs * numNodesPerLayer];
    float d_hidden_weights[numNodesPerLayer * numNodesPerLayer * (numHiddenLayers - 1)];
    float d_output_weights[numNodesPerLayer * numOutputs];

    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numNodesPerLayer; j++) {
            d_input_weights[i * numNodesPerLayer + j] = 0.0;
        }
    }
    for (int l = 0; l < numHiddenLayers - 1; l++) {
        for (int i = 0; i < numNodesPerLayer; i++) {
            for (int j = 0; j < numNodesPerLayer; j++) {
                d_hidden_weights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] = 0.0;
            }
        }
    }

   for (int i = 0; i < numNodesPerLayer; i++) {
        for (int j = 0; j < numOutputs; j++) {
            d_output_weights[i * numOutputs + j] = 0.0;
        }
    }

    float learning_rate = 0.00001;

    float d_output_nodes[numOutputs];
    float d_hidden_nodes[numNodesPerLayer * numHiddenLayers];
    
    // dE/dwij = dE/doj * doj/dwij = dE/dojk * doj/dnetj * dnetj / dwij
    // dnetj / dwij = oi
    // https://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error


    //Last layer of weights
    for (int j = 0; j < numOutputs; j++) {
        //if (outputVec[j] > 0) {
            d_output_nodes[j] = diff;
        //} else {
           // d_output_nodes[j] = 0;   
        //}
       
    }
    

   for (int i = 0; i < numNodesPerLayer; i++) {
        for (int j = 0; j < numOutputs; j++) {
            d_output_weights[i * numOutputs + j] = hiddenVec[(numHiddenLayers - 1) * numNodesPerLayer + i] * d_output_nodes[j];
        }
    }
        
    //Last hidden layer
    for (int j = 0; j < numNodesPerLayer; j++) {
        d_hidden_nodes[(numHiddenLayers - 1) * numNodesPerLayer + j] = 0.0;
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
            d_hidden_nodes[(l-1) * numNodesPerLayer + i] = 0.0;
            for (int j = 0; j < numNodesPerLayer; j++) {
                d_hidden_weights[(l-1) * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] = hiddenVec[(l-1) * numNodesPerLayer + i] * d_hidden_nodes[l * numNodesPerLayer + j];
                if (hiddenVec[(l-1) * numNodesPerLayer + i] > 0) {
                    d_hidden_nodes[(l-1) * numNodesPerLayer + i] += hiddenWeights[(l-1) * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] * d_hidden_nodes[l * numNodesPerLayer + j];
                }
            }
        }
    }
    
    //Input layer
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numNodesPerLayer; j++) {
            inputWeights[i * numNodesPerLayer + j] -= inputVec[i] * d_hidden_nodes[j] * learning_rate;
        }
    }
        
    
    //Add this ray sample to the final weight delta
    
    for (int l = 0; l < numHiddenLayers - 1; l++) {
        for (int i = 0; i < numNodesPerLayer; i++) {
            for (int j = 0; j < numNodesPerLayer; j++) {
                hiddenWeights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] -= d_hidden_weights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] * learning_rate;
            }
        }
    }
    
        
    for (int i = 0; i < numNodesPerLayer; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i * numOutputs + j] -= d_output_weights[i * numOutputs + j] * learning_rate;
        }
    }

    //updating bias weights
    for (int l = 0; l < numHiddenLayers; l++) {
        for (int i = 0; i < numNodesPerLayer; i++) {
            hiddenBiasWeights[l * numNodesPerLayer + i] += d_hidden_nodes[l * numNodesPerLayer + i] * learning_rate;
        }
    }
    for (int i = 0; i < numOutputs; i++) {
        outputBiasWeights[i] += d_output_nodes[i] * learning_rate;
    }
    
    
}


void main() {
    float value = forward(vec3(0.69, 0.420, 0.0));
    loss = (0.5 - value) * (0.5 - value);
    //df_output_weights[0] = 5;
    backwards(loss);
    
}