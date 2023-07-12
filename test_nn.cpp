#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <process.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <time.h>
#include "stb_image.h"
#include <stdio.h>
#include <stdlib.h>

const int windowWidth = 32;
const int windowHeight = 24;

std::string readShaderFiles(const std::string& filePath)
{
    std::ifstream file(filePath);
    if (!file)
    {
        std::cerr << "Failed to open shader file: " << filePath << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

double randf(double min, double max)
{
    return (((double)rand()) / RAND_MAX) * (max - min) + min;
}
struct pos {
    float x, y;
};


int main() {

    srand(time(NULL));

    if (!glfwInit()) {
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "NeRFGL", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    //glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glViewport(0, 0, windowWidth, windowHeight);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    if (glewInit() != GLEW_OK) {
        return -1;
    }

    std::string compSource = "#version 430 core\n";
    compSource.append(std::string("#define WORK_GROUP_SIZE_X ").append(std::to_string(windowWidth)).append("\n"));
    compSource.append(std::string("#define WORK_GROUP_SIZE_Y ").append(std::to_string(windowHeight)).append("\n"));
    compSource.append(std::string("#define WORK_GROUP_SIZE_Z ").append(std::to_string(1)).append("\n"));
    compSource.append(readShaderFiles("nn_compute_shader.glsl"));
    const char* computeShaderSource = compSource.c_str();

    GLint compileStatus;
    unsigned int computeShader;
    computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &computeShaderSource, NULL);
    glCompileShader(computeShader);

    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &compileStatus);
    if (compileStatus == GL_FALSE)
    {
        GLint logLength;
        glGetShaderiv(computeShader, GL_INFO_LOG_LENGTH, &logLength);
        std::string log(logLength, ' ');
        glGetShaderInfoLog(computeShader, logLength, nullptr, &log[0]);
        std::cerr << "Compute shader compilation failed: " << log << std::endl;
        glfwTerminate();
        return -1;
    }

    const unsigned int numHiddenLayers = 2;
    const unsigned int numNodesPerLayer = 20;
    const unsigned int numInputs = 2;
    const unsigned int numOutputs = 1;

    const int trainingSetSize = 5000;
    const int batchSize = 500;
    float learning_rate = 1;

    int numEpochs = 30000;

    float inputWeights[numInputs * numNodesPerLayer];
    float* batchDeltaInputWeights = (float*)malloc(sizeof(float)*numInputs * numNodesPerLayer * batchSize);

    float* hiddenWeights = (float*)malloc(sizeof(float) * (numHiddenLayers - 1) * numNodesPerLayer * numNodesPerLayer);
    float* batchDeltaHiddenWeights = (float*)malloc(sizeof(float) * (numHiddenLayers - 1) * numNodesPerLayer * numNodesPerLayer * batchSize);

    float outputWeights[numNodesPerLayer * numOutputs];
    float* batchDeltaOutputWeights = (float*)malloc(sizeof(float) * numNodesPerLayer * numOutputs * batchSize);

    float hiddenBias[numHiddenLayers * numNodesPerLayer];
    float outputBias[numOutputs];
    float* batchDeltaBiasWeights = (float*)malloc(sizeof(float) * (numHiddenLayers * numNodesPerLayer + numOutputs) * batchSize);

    struct pos inputData[trainingSetSize];
    float outputData[trainingSetSize];

    for (int i = 0; i < trainingSetSize; i++) {
        float x = randf(-2.0, 2.0);
        float y = randf(-2.0, 2.0);
        inputData[i].x = x;
        inputData[i].y = y;
        outputData[i] = (sin(3 * x) + 1.0 > y && x - 1.0 < y) ? 1.0 : 0.0;
    }

    //initialize weights

    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numNodesPerLayer; j++) {
            inputWeights[i * numNodesPerLayer + j] = randf(0, 1);
            
        }
    }
    for (int l = 0; l < numHiddenLayers; l++) {
        for (int i = 0; i < numNodesPerLayer; i++) {
            hiddenBias[l * numNodesPerLayer + i] = randf(0, 1.0);
        }
    }
    for (int l = 0; l < numHiddenLayers - 1; l++) {
        for (int i = 0; i < numNodesPerLayer; i++) {
            for (int j = 0; j < numNodesPerLayer; j++) {
                hiddenWeights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] = randf(0, 1);
            }
        }
    }
    for (int j = 0; j < numOutputs; j++) {
        outputBias[j] = randf(0.0, 1.0);
    }
    for (int i = 0; i < numNodesPerLayer; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i * numOutputs + j] = randf(0, 1);
        }
    }

    GLuint hiddenWeightsSSBO;
    glGenBuffers(1, &hiddenWeightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, hiddenWeightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer * (numHiddenLayers - 1) * numNodesPerLayer * sizeof(float), &hiddenWeights[0], GL_DYNAMIC_DRAW);

    GLuint inputWeightsSSBO;
    glGenBuffers(1, &inputWeightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputWeightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numInputs * numNodesPerLayer * sizeof(float), &inputWeights[0], GL_DYNAMIC_DRAW);

    GLuint outputWeightsSSBO;
    glGenBuffers(1, &outputWeightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputWeightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer * numOutputs * sizeof(float), &outputWeights[0], GL_DYNAMIC_DRAW);

    GLuint batchDeltaHiddenWeightsSSBO;
    glGenBuffers(1, &batchDeltaHiddenWeightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, batchDeltaHiddenWeightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer * numNodesPerLayer * (numHiddenLayers - 1) * batchSize * sizeof(float), &batchDeltaHiddenWeights[0], GL_DYNAMIC_DRAW);

    GLuint batchDeltaInputWeightsSSBO;
    glGenBuffers(1, &batchDeltaInputWeightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, batchDeltaInputWeightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numInputs * batchSize * numNodesPerLayer * sizeof(float), &batchDeltaInputWeights[0], GL_DYNAMIC_DRAW);

    GLuint batchDeltaOutputWeightsSSBO;
    glGenBuffers(1, &batchDeltaOutputWeightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, batchDeltaOutputWeightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numOutputs * numNodesPerLayer * batchSize * sizeof(float), &batchDeltaOutputWeights[0], GL_DYNAMIC_DRAW);

    GLuint batchDeltaBiasWeightsSSBO;
    glGenBuffers(1, &batchDeltaBiasWeightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, batchDeltaBiasWeightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (numHiddenLayers * numNodesPerLayer + numOutputs) * batchSize * sizeof(float), &batchDeltaBiasWeights[0], GL_DYNAMIC_DRAW);

    float loss[batchSize];
    GLuint lossSSBO;
    glGenBuffers(1, &lossSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, lossSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * batchSize, &loss, GL_DYNAMIC_DRAW);

    GLuint hiddenBiasSSBO;
    glGenBuffers(1, &hiddenBiasSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, hiddenBiasSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numHiddenLayers * numNodesPerLayer * sizeof(float), &hiddenBias[0], GL_DYNAMIC_DRAW);

    GLuint outputBiasSSBO;
    glGenBuffers(1, &outputBiasSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBiasSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numOutputs * sizeof(float), &outputBias[0], GL_DYNAMIC_DRAW);

    GLuint inputDataSSBO;
    glGenBuffers(1, &inputDataSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputDataSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, trainingSetSize * sizeof(pos), &inputData[0], GL_DYNAMIC_DRAW);

    GLuint outputDataSSBO;
    glGenBuffers(1, &outputDataSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputDataSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, trainingSetSize * sizeof(float), &outputData[0], GL_DYNAMIC_DRAW);

    GLuint computeProgram;
    computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);

    GLint indexLoc = glGetUniformLocation(computeProgram, "startTrainingIndex");

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, hiddenWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, inputWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, outputWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, batchDeltaHiddenWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, batchDeltaInputWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, batchDeltaOutputWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, lossSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, hiddenBiasSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, outputBiasSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, inputDataSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, outputDataSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, batchDeltaBiasWeightsSSBO);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);


    

    for (int epoch = 0; epoch < numEpochs; epoch++) {
        float totalLoss = 0.0;
        for (int index = 0; index < trainingSetSize - batchSize + 1; index += batchSize) {
            glUniform1i(indexLoc, index);

            glUseProgram(computeProgram);
            glDispatchCompute(1, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

            glGetNamedBufferSubData(inputWeightsSSBO, 0, numNodesPerLayer * numInputs * sizeof(float), &inputWeights[0]);
            glGetNamedBufferSubData(hiddenWeightsSSBO, 0, numNodesPerLayer * numNodesPerLayer * (numHiddenLayers - 1) * sizeof(float), &hiddenWeights[0]);
            glGetNamedBufferSubData(outputWeightsSSBO, 0, numNodesPerLayer * numOutputs * sizeof(float), &outputWeights[0]);

            glGetNamedBufferSubData(batchDeltaInputWeightsSSBO, 0, batchSize * numNodesPerLayer * numInputs * sizeof(float), &batchDeltaInputWeights[0]);
            glGetNamedBufferSubData(batchDeltaHiddenWeightsSSBO, 0, batchSize * numNodesPerLayer * numNodesPerLayer * (numHiddenLayers - 1) * sizeof(float), &batchDeltaHiddenWeights[0]);
            glGetNamedBufferSubData(batchDeltaOutputWeightsSSBO, 0, batchSize * numNodesPerLayer * numOutputs * sizeof(float), &batchDeltaOutputWeights[0]);

            //glGetNamedBufferSubData(inputVecSSBO, 0, numInputs * sizeof(float), &inputVec[0]);
            //glGetNamedBufferSubData(hiddenVecSSBO, 0, numNodesPerLayer * numHiddenLayers * sizeof(float), &hiddenVec[0]);
            //glGetNamedBufferSubData(outputVecSSBO, 0, numOutputs * sizeof(float), &outputVec[0]);
            glGetNamedBufferSubData(hiddenBiasSSBO, 0, numNodesPerLayer * numHiddenLayers * sizeof(float), &hiddenBias[0]);
            glGetNamedBufferSubData(outputBiasSSBO, 0, numOutputs * sizeof(float), &outputBias[0]);
            glGetNamedBufferSubData(batchDeltaBiasWeightsSSBO, 0, (numHiddenLayers * numNodesPerLayer + numOutputs) * batchSize * sizeof(float), &batchDeltaBiasWeights[0]);

            glGetNamedBufferSubData(lossSSBO, 0, sizeof(float) * batchSize, &loss);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

            //Update weights
            for (int k = 0; k < batchSize; k++) {
                //std::cout << loss[k] << std::endl;
                totalLoss += loss[k];

            }
            for (int i = 0; i < numInputs; i++) {
                for (int j = 0; j < numNodesPerLayer; j++) {
                    for (int k = 0; k < batchSize; k++) {
                        inputWeights[i * numNodesPerLayer + j] -= batchDeltaInputWeights[(i * numNodesPerLayer + j) * batchSize + k] * learning_rate;
                    }
                }
            }
            for (int l = 0; l < numHiddenLayers - 1; l++) {
                for (int i = 0; i < numNodesPerLayer; i++) {
                    for (int j = 0; j < numNodesPerLayer; j++) {
                        for (int k = 0; k < batchSize; k++) {
                            hiddenWeights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] -= batchDeltaHiddenWeights[(l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j) * batchSize + k] * learning_rate;
                            //std::cout << batchDeltaHiddenWeights[(l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j) * batchSize + k] << " ";
                        }
                    }
                }
            }
            for (int i = 0; i < numNodesPerLayer; i++) {
                for (int j = 0; j < numOutputs; j++) {
                    for (int k = 0; k < batchSize; k++) {
                        outputWeights[i * numOutputs + j] -= batchDeltaOutputWeights[(i * numOutputs + j) * batchSize + k] * learning_rate;
                        //std::cout << batchDeltaOutputWeights[(i * numOutputs + j) * batchSize + k] << " ";
                    }
                }
            }
            for (int k = 0; k < batchSize; k++) {
                for (int l = 0; l < numHiddenLayers; l++) {
                    for (int i = 0; i < numNodesPerLayer; i++) {
                        hiddenBias[l * numNodesPerLayer + i] -= batchDeltaBiasWeights[(l * numNodesPerLayer + i) * batchSize + k] * learning_rate;
                    }
                }
                for (int i = 0; i < numOutputs; i++) {
                    outputBias[i] -= batchDeltaBiasWeights[(numHiddenLayers * numNodesPerLayer + i) * batchSize + k] * learning_rate;
                }
            }
            /*
            std::cout << "Input Vec: ";
            for (int i = 0; i < numInputs; i++) {
                std::cout << inputVec[i] << " ";
            }

            for (int l = 0; l < numHiddenLayers; l++) {
                std::cout << std::endl << "Hidden Layer " << l + 1 << " ";
                for (int i = 0; i < numNodesPerLayer; i++) {
                    std::cout << i << "=" << hiddenVec[l * numNodesPerLayer + i] << " ";
                }
            }
            std::cout << std::endl << "Output Layer ";
            for (int i = 0; i < numOutputs; i++) {
                std::cout << outputVec[i] << " ";
            }
            std::cout << std::endl;

            for (int l = 0; l < numHiddenLayers; l++) {
                std::cout << std::endl << "Hidden Bias " << l + 1 << " ";
                for (int i = 0; i < numNodesPerLayer; i++) {
                    std::cout << i << "=" << hiddenBias[l * numNodesPerLayer + i] << " ";
                }
            }
            std::cout << std::endl << "Output Bias ";
            for (int i = 0; i < numOutputs; i++) {
                std::cout << outputBias[i] << " ";
            }
            std::cout << std::endl;

            for (int i = 0; i < numInputs; i++) {
                for (int j = 0; j < numNodesPerLayer; j++) {
                    //std::cout << inputWeights[i * numNodesPerLayer + j] << " ";
                }
            }
            for (int l = 0; l < numHiddenLayers - 1; l++) {
                for (int i = 0; i < numNodesPerLayer; i++) {
                    for (int j = 0; j < numNodesPerLayer; j++) {
                        //std::cout << hiddenWeights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] << " ";
                    }
                }
            }
            for (int i = 0; i < numNodesPerLayer; i++) {
                for (int j = 0; j < numOutputs; j++) {
                    std::cout << outputWeights[i * numOutputs  + j] << " ";
                }
            }
            */

            /*
            for (int i = 0; i < numInputs; i++) {
                for (int j = 0; j < numNodesPerLayer; j++) {
                    inputWeights[i * numNodesPerLayer + j] -= DFinputWeights[i * numNodesPerLayer + j];
                }
            }
            for (int l = 0; l < numHiddenLayers - 1; l++) {
                for (int i = 0; i < numNodesPerLayer; i++) {
                    for (int j = 0; j < numNodesPerLayer; j++) {
                        hiddenWeights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] -= DFhiddenWeights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j];
                    }
                }
            }
            for (int i = 0; i < numNodesPerLayer; i++) {
                for (int j = 0; j < numOutputs; j++) {
                    outputWeights[i * numOutputs + j] -= DFoutputWeights[i * numOutputs + j];
                }
            }
            */




            //Moved updated weights into GPU memory

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, hiddenWeightsSSBO);
            glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer * (numHiddenLayers - 1) * numNodesPerLayer * sizeof(float), &hiddenWeights[0], GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputWeightsSSBO);
            glBufferData(GL_SHADER_STORAGE_BUFFER, numInputs * numNodesPerLayer * sizeof(float), &inputWeights[0], GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputWeightsSSBO);
            glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer * numOutputs * sizeof(float), &outputWeights[0], GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, hiddenBiasSSBO);
            glBufferData(GL_SHADER_STORAGE_BUFFER, numHiddenLayers * numNodesPerLayer * sizeof(float), &hiddenBias[0], GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBiasSSBO);
            glBufferData(GL_SHADER_STORAGE_BUFFER, numOutputs * sizeof(float), &outputBias[0], GL_DYNAMIC_DRAW);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
        }
        std::cout << "Loss " << totalLoss / trainingSetSize << std::endl;
    }
}