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

    const unsigned int numHiddenLayers = 5;
    const unsigned int numNodesPerLayer = 5;
    const unsigned int numInputs = 3;
    const unsigned int numOutputs = 1;


    float inputWeights[numInputs * numNodesPerLayer];
    float inputVec[numInputs];
    float* hiddenWeights = (float*)malloc(sizeof(float) * (numHiddenLayers - 1) * numNodesPerLayer * numNodesPerLayer);
    float* hiddenVec = (float*)malloc(sizeof(float) * numHiddenLayers * numNodesPerLayer);
    float outputWeights[numNodesPerLayer * numOutputs];
    float outputVec[numOutputs];
    float hiddenBias[numHiddenLayers * numNodesPerLayer];
    float outputBias[numOutputs];

    //initialize weights

    for (int i = 0; i < numInputs; i++) {
        inputVec[i] = 0.0;
        for (int j = 0; j < numNodesPerLayer; j++) {
            inputWeights[i * numNodesPerLayer + j] = randf(0, 1);
            
        }
    }
    for (int l = 0; l < numHiddenLayers; l++) {
        for (int i = 0; i < numNodesPerLayer; i++) {
            hiddenBias[l * numNodesPerLayer + i] = randf(0, 1.0);
            hiddenVec[l * numNodesPerLayer + i] = 0;
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
        outputVec[j] = 0.0;
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

    GLuint hiddenVecSSBO;
    glGenBuffers(1, &hiddenVecSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, hiddenVecSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer * numHiddenLayers  * sizeof(float), &hiddenVec[0], GL_DYNAMIC_DRAW);

    GLuint inputVecSSBO;
    glGenBuffers(1, &inputVecSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputVecSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numInputs * sizeof(float), &inputVec[0], GL_DYNAMIC_DRAW);

    GLuint outputVecSSBO;
    glGenBuffers(1, &outputVecSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputVecSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numOutputs * sizeof(float), &outputVec[0], GL_DYNAMIC_DRAW);

    float loss;
    GLuint lossSSBO;
    glGenBuffers(1, &lossSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, lossSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float), &loss, GL_DYNAMIC_DRAW);

    GLuint hiddenBiasSSBO;
    glGenBuffers(1, &hiddenBiasSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, hiddenBiasSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numHiddenLayers * numNodesPerLayer * sizeof(float), &hiddenBias[0], GL_DYNAMIC_DRAW);

    GLuint outputBiasSSBO;
    glGenBuffers(1, &outputBiasSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBiasSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numOutputs * sizeof(float), &outputBias[0], GL_DYNAMIC_DRAW);

    GLuint computeProgram;
    computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, hiddenWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, inputWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, outputWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, hiddenVecSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, inputVecSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, outputVecSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, lossSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, hiddenBiasSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, outputBiasSSBO);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

    for (int i = 0; i < 20; i++) {

        glUseProgram(computeProgram);
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

        //glGetNamedBufferSubData(inputWeightsSSBO, 0, numNodesPerLayer * numInputs * sizeof(float), &inputWeights[0]);
        //glGetNamedBufferSubData(hiddenWeightsSSBO, 0, numNodesPerLayer * numNodesPerLayer * (numHiddenLayers - 1) * sizeof(float), &hiddenWeights[0]);
        //glGetNamedBufferSubData(outputWeightsSSBO, 0, numNodesPerLayer * numOutputs * sizeof(float), &outputWeights[0]);
        glGetNamedBufferSubData(inputVecSSBO, 0, numInputs * sizeof(float), &inputVec[0]);
        glGetNamedBufferSubData(hiddenVecSSBO, 0, numNodesPerLayer * numHiddenLayers * sizeof(float), &hiddenVec[0]);
        glGetNamedBufferSubData(outputVecSSBO, 0, numOutputs * sizeof(float), &outputVec[0]);
        glGetNamedBufferSubData(hiddenBiasSSBO, 0, numNodesPerLayer* numHiddenLayers * sizeof(float), &hiddenBias[0]);
        glGetNamedBufferSubData(outputBiasSSBO, 0, numOutputs * sizeof(float), &outputBias[0]);
        glGetNamedBufferSubData(lossSSBO, 0,  sizeof(float), &loss);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
       
        //Update weights
        std::cout << "Loss " << loss << std::endl;
        
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
                std::cout << inputWeights[i * numNodesPerLayer + j] << " ";
            }
        }
        for (int l = 0; l < numHiddenLayers - 1; l++) {
            for (int i = 0; i < numNodesPerLayer; i++) {
                for (int j = 0; j < numNodesPerLayer; j++) {
                    std::cout << hiddenWeights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] << " ";
                }
            }
        }
        for (int i = 0; i < numNodesPerLayer; i++) {
            for (int j = 0; j < numOutputs; j++) {
                std::cout << outputWeights[i * numOutputs  + j] << " ";
            }
        }
        
        
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
        /*
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, hiddenWeightsSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer * (numHiddenLayers - 1) * numNodesPerLayer * sizeof(float), &hiddenWeights[0], GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputWeightsSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, numInputs * numNodesPerLayer * sizeof(float), &inputWeights[0], GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputWeightsSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer * numOutputs * sizeof(float), &outputWeights[0], GL_DYNAMIC_DRAW);
        */
    }
}