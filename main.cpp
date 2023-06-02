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

int windowWidth = 800;
int windowHeight = 600;


void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    windowWidth = width;
    windowHeight = height;

    glViewport(0, 0, width, height);
}

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

int main() {

    srand(time(NULL));

    if (!glfwInit()) {
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
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
    compSource.append(std::string("#define WORK_GROUP_SIZE_X ").append(std::to_string(workGroupSize)).append("\n"));
    compSource.append(readShaderFile("compute_shader.glsl"));
    const char* computeShaderSource = compSource.c_str();


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


    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
    glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 cameraDirection = glm::normalize(cameraPos - cameraTarget);

    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 cameraRight = glm::normalize(glm::cross(up, cameraDirection));

    glm::vec3 cameraUp = glm::cross(cameraDirection, cameraRight);

    const float radius = 10.0f;

    const unsigned int numHiddenLayers = 3;
    const unsigned int numNodesPerLayer = 10;
    const unsigned int numInputs = 6;
    const unsigned int numOutputs = 4;

    float inputWeights[numInputs * numNodesPerLayer];
    float *hiddenWeights = (float*)malloc(sizeof(float) * (numHiddenLayers - 1) * numNodesPerLayer * numNodesPerLayer);
    float outputWeights[numNodesPerLayer * numOutputs];

    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numNodesPerLayer; j++) {
            inputWeights[i * numNodesPerLayer + j] = ranf(-1, 1);
        }
    }
    for (int l = 0; l < numHiddenLayers - 1; l++) {
        for (int i = 0; i < numNodesPerLayer; i++) {
            for (int j = 0; j < numNodesPerLayer; j++) {
                hiddenWeights[l * numNodesPerLayer * numNodesPerLayer + i * numNodesPerLayer + j] = ranf(-1, 1);
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
    glBufferData(GL_SHADER_STORAGE_BUFFER, numInputs * numNodesPerLayer * sizeof(float), &hiddenWeights[0], GL_DYNAMIC_DRAW);

    GLuint inputWeightsSSBO;
    glGenBuffers(1, &inputWeightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputWeightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numHiddenLayers * numNodesPerLayer * sizeof(float), &inputWeights[0], GL_DYNAMIC_DRAW);

    GLuint outputWeightsSSBO;
    glGenBuffers(1, &outputWeightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputWeightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numNodesPerLayer * numOutputs * sizeof(float), &outputWeights[0], GL_DYNAMIC_DRAW);

    
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
    GLint timeLoc = glGetUniformLocation(shaderProgram, "time");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteShader(computeShader);



    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, hiddenWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, inputWeightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, outputWeightsSSBO);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

    float lastFrameTime = static_cast<float>(glfwGetTime());

    glViewport(0, 0, windowWidth, windowHeight);
    while (!glfwWindowShouldClose(window)) {

        frameCount++;

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glUniform2f(resolutionLoc, static_cast<float>(width), static_cast<float>(height));

        float time = static_cast<float>(glfwGetTime());
        glUniform1f(timeLoc, time);

        glClear(GL_COLOR_BUFFER_BIT);

        if (frameCount % 100 == 0) {
            float fps = 1 / (time - lastFrameTime);
            std::string fpsText = "FPS: " + std::to_string(fps);
            std::cout << fpsText << std::endl;
        }
        
        lastFrameTime = time;

        glUseProgram(computeProgram);
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

        glUseProgram(shaderProgram);

        
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, 0, vertices);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDrawArrays(GL_TRIANGLES, 3, 3);
        glDisableClientState(GL_VERTEX_ARRAY);


        glfwSwapBuffers(window);
        glfwPollEvents();
    }

}