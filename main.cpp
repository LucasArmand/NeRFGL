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
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdio.h>
#include <stdlib.h>
const int windowWidth = 32;
const int windowHeight = 24;

/*
void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    windowWidth = width;
    windowHeight = height;

    glViewport(0, 0, width, height);
}
*/

std::string readShaderFile(const std::string& filePath)
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
struct pos {
    float x, y, z, w;
};

float vertices[] = {
    -1.0f,  -1.0f, 0.0f,
    1.0f, 1.0f, 0.0f,
   -1.0f, 1.0f, 0.0f,
   -1.0f,  -1.0f, 0.0f,
    1.0f, 1.0f, 0.0f,
   1.0f, -1.0f, 0.0f,

};

double ranf(double min, double max)
{
    return (((double)rand()) / RAND_MAX) * (max - min) + min;
}



int main_inactive() {

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



    const unsigned int workGroupSize = 256;


    std::string vertSource = readShaderFile("vertex_shader.glsl");
    const char* vertexShaderSource = vertSource.c_str();

    std::string fragSource = readShaderFile("fragment_shader.glsl");
    const char* fragmentShaderSource = fragSource.c_str();

    std::string compSource = "#version 430 core\n";
    compSource.append(std::string("#define WORK_GROUP_SIZE_X ").append(std::to_string(windowWidth)).append("\n"));
    compSource.append(std::string("#define WORK_GROUP_SIZE_Y ").append(std::to_string(windowHeight)).append("\n"));
    compSource.append(std::string("#define WORK_GROUP_SIZE_Z ").append(std::to_string(1)).append("\n"));
    compSource.append(readShaderFile("compute_shader.glsl"));
    const char* computeShaderSource = compSource.c_str();

    const char* filename = "red.jpg"; // Replace with the path to your image file
    int width, height, channels;
    unsigned char* imageData = stbi_load(filename, &width, &height, &channels, 0);

    if (imageData == nullptr) {
        // Image loading failed
        // Handle the error
    }

    int frameCount = 0;
    float elapsedTime = 0.0f;
    float fps = 0.0f;

    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLint compileStatus;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &compileStatus);
    if (compileStatus == GL_FALSE)
    {
        GLint logLength;
        glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &logLength);
        std::string log(logLength, ' ');
        glGetShaderInfoLog(vertexShader, logLength, nullptr, &log[0]);
        std::cerr << "Vertex shader compilation failed: " << log << std::endl;
        glfwTerminate();
        return -1;
    }

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &compileStatus);
    if (compileStatus == GL_FALSE)
    {
        GLint logLength;
        glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &logLength);
        std::string log(logLength, ' ');
        glGetShaderInfoLog(fragmentShader, logLength, nullptr, &log[0]);
        std::cerr << "Fragment shader compilation failed: " << log << std::endl;
        //glfwTerminate();
        //return -1;
    }

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

    unsigned int texture;
    glGenTextures(1, &texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, imageData);
    glGenerateMipmap(GL_TEXTURE_2D);


    const unsigned int numHiddenLayers = 4;
    const unsigned int numNodesPerLayer = 40;
    const unsigned int numInputs = 6;
    const unsigned int numOutputs = 4;

    float inputWeights[numInputs * numNodesPerLayer];
    float DFinputWeights[numInputs * numNodesPerLayer];
    float *hiddenWeights = (float*)malloc(sizeof(float) * (numHiddenLayers - 1) * numNodesPerLayer * numNodesPerLayer);
    float *DFhiddenWeights = (float*)malloc(sizeof(float) * (numHiddenLayers - 1) * numNodesPerLayer * numNodesPerLayer);
    float outputWeights[numNodesPerLayer * numOutputs];
    float DFoutputWeights[numNodesPerLayer * numOutputs];

    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numNodesPerLayer; j++) {
            inputWeights[i * numNodesPerLayer + j] = ranf(-1, 1);
        }
    }
    for (int l = 0; l < numHiddenLayers - 1; l++) {
        for (int i = 0; i < numNodesPerLayer; i++) {
            for (int j = 0; j < numNodesPerLayer; j++) {
                hiddenWeights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] = ranf(-1, 1);
                DFhiddenWeights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] = 0.0;
            }
        }
    }
    for (int i = 0; i < numNodesPerLayer; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i * numOutputs + j] = ranf(-1, 1);
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

    GLuint DFhiddenWeightsSSBO;
    glGenBuffers(1, &DFhiddenWeightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DFhiddenWeightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer * (numHiddenLayers - 1) * numNodesPerLayer * sizeof(float), &DFhiddenWeights[0], GL_DYNAMIC_DRAW);

    GLuint DFinputWeightsSSBO;
    glGenBuffers(1, &DFinputWeightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DFinputWeightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numInputs * numNodesPerLayer * sizeof(float), &DFinputWeights[0], GL_DYNAMIC_DRAW);

    GLuint DFoutputWeightsSSBO;
    glGenBuffers(1, &DFoutputWeightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, DFoutputWeightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer* numOutputs * sizeof(float), &DFoutputWeights[0], GL_DYNAMIC_DRAW);

    
    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    GLuint computeProgram;
    computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);

    GLint resolutionLoc = glGetUniformLocation(shaderProgram, "resolution");
    GLint resolutionCompLoc = glGetUniformLocation(computeProgram, "resolution");
    GLint timeLoc = glGetUniformLocation(shaderProgram, "time");
    GLint timeCompLoc = glGetUniformLocation(computeProgram, "time");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteShader(computeShader);
    


    /*
    unsigned int framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    unsigned int frameTexture;
    glGenTextures(1, &frameTexture);
    glBindTexture(GL_TEXTURE_2D, frameTexture);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, &textureData[0]);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    GLuint depthrenderbuffer;
    glGenRenderbuffers(1, &depthrenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, windowWidth, windowHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, frameTexture, 0);
    GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, DrawBuffers);
    */
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        return false;
    }

    

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, hiddenWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, inputWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, outputWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, DFhiddenWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, DFinputWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, DFoutputWeightsSSBO);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

    float lastFrameTime = static_cast<float>(glfwGetTime());
    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer); //?
    glViewport(0, 0, windowWidth, windowHeight);
    char* textureData = (char*)malloc(3 * windowWidth * windowHeight * sizeof(char));
    while (!glfwWindowShouldClose(window)) {

        frameCount++;

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glUniform2f(resolutionLoc, static_cast<float>(width), static_cast<float>(height));
        glUniform2f(resolutionCompLoc, static_cast<float>(width), static_cast<float>(height));
        float time = static_cast<float>(glfwGetTime());
        glUniform1f(timeLoc, time);
        glUniform1f(timeCompLoc, time);

        glClear(GL_COLOR_BUFFER_BIT);

        if (frameCount % 5 == 0) {
            float fps = 1 / (time - lastFrameTime);
            std::string fpsText = "FPS: " + std::to_string(fps);
            std::cout << fpsText << std::endl;
        }
        
        lastFrameTime = time;

        glUseProgram(computeProgram);
        glDispatchCompute(windowWidth, windowHeight, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

        glGetNamedBufferSubData(DFinputWeightsSSBO, 0, numNodesPerLayer * numInputs * sizeof(float), &DFinputWeights[0]);
        glGetNamedBufferSubData(DFhiddenWeightsSSBO, 0, numNodesPerLayer * numNodesPerLayer * (numHiddenLayers - 1) * sizeof(float), &DFhiddenWeights[0]);
        glGetNamedBufferSubData(DFoutputWeightsSSBO, 0, numNodesPerLayer * numOutputs * sizeof(float), &DFoutputWeights[0]);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

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

        glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer * (numHiddenLayers - 1) * numNodesPerLayer * sizeof(float), &hiddenWeights[0], GL_DYNAMIC_DRAW);
        glBufferData(GL_SHADER_STORAGE_BUFFER, numInputs * numNodesPerLayer * sizeof(float), &inputWeights[0], GL_DYNAMIC_DRAW);
        glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer * numOutputs * sizeof(float), &outputWeights[0], GL_DYNAMIC_DRAW);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
        glUseProgram(shaderProgram);
        //std::cout << std::endl;
        //for (int i = 0; i < 5; i++) {
        //    for (int j = 0; j < numOutputs; j++) {
        //        std::cout << DFoutputWeights[i * numOutputs + j] << " ";
        //    }
        //}
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, 0, vertices);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDrawArrays(GL_TRIANGLES, 3, 3);
        glDisableClientState(GL_VERTEX_ARRAY);

        /*
        glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, &textureData[0]);
        for (int x = 0; x < windowWidth; x++) {
            for (int y = 0; y < windowHeight; y++) {
                int pixelIndex = (y * width + x) * 3;
                int d_red = imageData[pixelIndex] - textureData[pixelIndex];
                int d_green = imageData[pixelIndex + 1] - textureData[pixelIndex + 1];
                int d_blue = imageData[pixelIndex + 2] - textureData[pixelIndex + 2];
                //std::cout << "Loss for " << x << ", " << y << ": (" << d_red << ", " << d_green << ", " << d_blue << ")" << std::endl;
            }
        }
        */
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

}