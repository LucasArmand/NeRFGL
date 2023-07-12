// The following defines are set in main.cpp
// #define WORK_GROUP_SIZE_X
#define BATCH_SIZE 500

layout (local_size_x = BATCH_SIZE, local_size_y = 1, local_size_z = 1) in;
    float rate = 0.000001;
//layout(std430, binding=5) buffer Pos{
//    vec4 positions[];
//};
layout(std430, binding=3) buffer HiddenWeights{
    float hw[];
};
layout(std430, binding=4) buffer InputWeights{
    float iw[];
};
layout(std430, binding=5) buffer OutputWeights{
    float ow[];
};
layout(std430, binding=6) buffer BatchDeltaHiddenWeights{
    float bdhw[];
};
layout(std430, binding=7) buffer BatchDeltaInputWeights{
    float bdiw[];
};
layout(std430, binding=8) buffer BatchDeltaOutputWeights{
    float bdow[];
};
layout(std430, binding=14) buffer BatchDeltaBiasWeights{
    float bdbw[];
};
layout(std430, binding=10) buffer HiddenBiasWeights{
    float hb[];
};
layout(std430, binding=11) buffer OutputBiasWeights{
    float ob[];
};
layout(std430, binding=9) buffer Loss{
    float loss[];
};
layout(std430, binding=12) buffer InputData{
    vec2 inputData[];
};
layout(std430, binding=13) buffer OutputData{
    float outputData[];
};

uniform vec2 resolution;
uniform float time;

uniform sampler2D imageTexture;

vec3 sampleColor;
float sampleDensity;

const int trainingSetSize = 5000;
uniform int startTrainingIndex;

const unsigned int layers = 2;
const unsigned int hSize = 20;
const unsigned int iSize = 2;
const unsigned int oSize = 1;
const unsigned int batchSize = BATCH_SIZE;
unsigned int batch_index;

float iVec[iSize];
float hVec[hSize * layers];
float oVec[oSize];

const int MAX_ITER = 5;

float targetVec[oSize];
int trainingIndex = 0;

struct raySample {
    vec3 pos;
    vec3 dir;
    vec3 color;
    float density;
    float alpha;

};

struct raySample samples[MAX_ITER];

float forward(vec2 pos) {

    iVec[0] = pos.x;
    iVec[1] = pos.y;

    // Initialize hidden nodes
    for (int i = 0; i < layers; i++) {
        for (int j = 0; j < hSize; j++) {
            hVec[i * hSize + j] = 0.0;
        }
    }

    // Initialize output nodes
    for (int i = 0; i < oSize; i++) {
        oVec[i] = 0.0;
    }

    //Forward to first hidden layer
    for (int j = 0; j < hSize; j++) {
        for (int i = 0; i < iSize; i++) {
            hVec[j] += iVec[i] * iw[i * hSize + j];
        }
        hVec[j] += hb[j];
     }
    

    for(int i = 0; i < hSize; i++) {

        // ReLU
        hVec[i] = max(0, hVec[i]);

        // sigmoid
        //hVec[i] = 1 / (1 + exp(-hVec[i]));
    }

    // Forward to l-th hidden layer
    for (int l = 1; l < layers; l++) {
        for (int j = 0; j < hSize; j++) {
            for (int i = 0; i < hSize; i++) {
                hVec[l * hSize + j] += hVec[(l-1) * hSize + i] * hw[(l-1) * hSize * hSize + i * hSize + j];
            }
            hVec[l * hSize + j] += hb[l * hSize + j];
        }

        // l-th Hidden activation function
        for(int i = 0; i < hSize; i++) {

            //RELU
            hVec[l * hSize + i] = max(0, hVec[l * hSize + i]);

            // sigmoid
            //hVec[l * hSize + i] = 1 / (1 + exp(-hVec[l * hSize + i]));

        }
    }

    // Forward to output layer
    for (int j = 0; j < oSize; j++) {
        for (int i = 0; i < hSize; i++) {
            oVec[j] += hVec[(layers - 1) * hSize + i] * ow[i * oSize + j];
        }
        oVec[j] += ob[j];
    }

    // Output activation function
    for(int i = 0; i < oSize; i++) {

        //ReLU
        //oVec[i] = max(0, oVec[i]);

        //sigmoid
        //oVec[i] = 1 / (1 + exp(-oVec[i]));

    }
    
    return oVec[0];

}



void backwards(float diff) {

    float d_iw[iSize * hSize];
    float d_hw[hSize * hSize * (layers - 1)];
    float d_ow[hSize * oSize];

    for (int i = 0; i < iSize; i++) {
        for (int j = 0; j < hSize; j++) {
            d_iw[i * hSize + j] = 0.0;
        }
    }
    for (int l = 0; l < layers - 1; l++) {
        for (int i = 0; i < hSize; i++) {
            for (int j = 0; j < hSize; j++) {
                d_hw[l * hSize * hSize + i * hSize + j] = 0.0;
            }
        }
    }

   for (int i = 0; i < hSize; i++) {
        for (int j = 0; j < oSize; j++) {
            d_ow[i * oSize + j] = 0.0;
        }
    }



    float d_oVec[oSize];
    float d_hVec[hSize * layers];


    // https://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error

    //Last layer of weights
    for (int j = 0; j < oSize; j++) {
         d_oVec[j] = diff;
    }
    

   for (int i = 0; i < hSize; i++) {
        for (int j = 0; j < oSize; j++) {
            d_ow[i * oSize + j] = hVec[(layers - 1) * hSize + i] * d_oVec[j];
        }
    }
        
    //Last hidden layer
    for (int j = 0; j < hSize; j++) {
        d_hVec[(layers - 1) * hSize + j] = 0.0;
        //ReLU derivative
        if (hVec[(layers - 1) * hSize + j] > 0) {
            for (int k = 0; k < oSize; k++) {
                d_hVec[(layers - 1) * hSize + j] += d_oVec[k] * ow[j * oSize + k];
            }
        }
    }
        
    //move backwards in hidden layers (destination, not including first layer)
    for (uint l = layers - 1; l > 0; l--) {
        for (int i = 0; i < hSize; i++) {
            d_hVec[(l-1) * hSize + i] = 0.0;
            for (int j = 0; j < hSize; j++) {
                d_hw[(l-1) * hSize * hSize + i * hSize + j] = hVec[(l-1) * hSize + i] * d_hVec[l * hSize + j];
                if (hVec[(l-1) * hSize + i] > 0) {
                    d_hVec[(l-1) * hSize + i] += hw[(l-1) * hSize * hSize + i * hSize + j] * d_hVec[l * hSize + j];
                }
            }
        }
    }
    
    //Input weights derivative
    for (int i = 0; i < iSize; i++) {
        for (int j = 0; j < hSize; j++) {
            d_iw[i * hSize + j] = iVec[i] * d_hVec[j];
        }
    }

    for (int i = 0; i < iSize; i++) {
        for (int j = 0; j < hSize; j++) {
            bdiw[(i * hSize + j) * batchSize + batch_index] = d_iw[i * hSize + j] * rate;
        }
    }
        
    for (int l = 0; l < layers - 1; l++) {
        for (int i = 0; i < hSize; i++) {
            for (int j = 0; j < hSize; j++) {
                bdhw[(l * hSize * hSize + i * hSize + j) * batchSize + batch_index] = d_hw[l * hSize * hSize + i * hSize + j] * rate;
            }
        }
    }
    
        
    for (int i = 0; i < hSize; i++) {
        for (int j = 0; j < oSize; j++) {
            bdow[(i * oSize + j) * batchSize + batch_index] = d_ow[i * oSize + j] * rate;
        }
    }

    //updating bias weights
    for (int l = 0; l < layers; l++) {
        for (int i = 0; i < hSize; i++) {
            bdbw[(l * hSize + i) * batchSize + batch_index] = d_hVec[l * hSize + i] * rate;
        }
    }
    for (int i = 0; i < oSize; i++) {
        bdbw[(layers * hSize + i) * batchSize + batch_index] = d_oVec[i] * rate;
    }
    
    
}


void main() {
    for (int i = 0; i < iSize; i++) {
        iVec[i] = 0.0;
    }
    for (int l = 0; l < layers; l++) {
        for (int i = 0; i < hSize; i++) {
            hVec[l * hSize + i] = 0.0;
        }
    }
    for (int i = 0; i < oSize; i++) {
        oVec[i] = 0.0;
    }
    batch_index = gl_GlobalInvocationID.x;
    loss[batch_index] = 0.0;

    float value = forward(inputData[startTrainingIndex + batch_index]);
    float target = outputData[startTrainingIndex + batch_index];
    backwards(value - target);

 
    //float value = forward(inputData[startTrainingIndex+ batch_index]);
    //float target = outputData[startTrainingIndex + batch_index];
    float err = (target - value) * (target - value);
    loss[batch_index] = err;
    //}
    //loss = loss / 2000;
}