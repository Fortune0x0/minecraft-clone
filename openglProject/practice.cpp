//
//#define _USE_MATH_DEFINES
//#include <cmath>
//#include <GL/glew.h>
//#include <GLFW/glfw3.h>
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include "stb_image.h"
//
//
//unsigned int getShader(const std::string& shaderSrc, unsigned int shaderType) {
//	unsigned int shader = glCreateShader(shaderType);
//	const char* src = shaderSrc.c_str();
//	glShaderSource(shader, 1, &src, nullptr);
//
//	glCompileShader(shader);
//	int result;
//	glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
//
//	if (result != GL_TRUE) {
//		int length;
//		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
//		char* message = new char[length];
//
//		glGetShaderInfoLog(shader, length, nullptr, message);
//		std::cerr << "SHADER ERROR >>>>>>>\nERROR OCCURED WITH" << ((shaderType == GL_VERTEX_SHADER) ? "VERTEX" : "FRAGMENT") << " SHADER\nERROR MEESAGE: " << message << "\n";
//		delete[] message;
//		glDeleteShader(shader);
//		return 0;
//	}
//
//	return shader;
//
//
//}
//unsigned int loadShaders(const std::string& vertexShader, const std::string& fragementShader) {
//	unsigned int program = glCreateProgram();
//	unsigned int vs = getShader(vertexShader, GL_VERTEX_SHADER);
//	unsigned int fs = getShader(fragementShader, GL_FRAGMENT_SHADER);
//
//	glAttachShader(program, vs);
//	glAttachShader(program, fs);
//	glLinkProgram(program);
//	glValidateProgram(program);
//
//	glDeleteShader(vs);
//	glDeleteShader(fs);
//
//	return program;
//
//
//}
//unsigned int getShaderProgram(const std::string& filepath) {
//	std::ifstream file(filepath);
//	if (!file.is_open()) {
//		std::cerr << "COULD NOT FIND SHADER FILE\n";
//		return 0;
//	}
//
//	enum class MODE { VERTEX = 0, FRAGMENT = 1 };
//	MODE mode;
//	std::string line;
//	std::stringstream s[2];
//
//	while (std::getline(file, line)) {
//		if (line.find("VERTEX") != std::string::npos) {
//			mode = MODE::VERTEX;
//		}
//		else if (line.find("FRAGEMENT") != std::string::npos) {
//			mode = MODE::FRAGMENT;
//		}
//
//		else {
//			s[(int)mode] << line << "\n";
//		}
//
//	}
//
//	return loadShaders(s[0].str(), s[1].str());
//
//}
//
//
//
//
//
//void setAsIdentityMat(float* matrix) {
//	for (int i = 0; i < 16; ++i) {
//		if (i % 5 == 0) {
//			matrix[i] = 1;
//		}
//		else {
//			matrix[i] = 0;
//		}
//	}
//}
//
//void rotateX(float* matrix, float angle) {
//	matrix[5] = std::cos(angle);
//	matrix[6] = -std::sin(angle);
//	matrix[9] = std::sin(angle);
//	matrix[10] = std::cos(angle);
//
//}
//
//void rotateY(float* matrix, float angle) {
//	matrix[0] = std::cos(angle);
//	matrix[2] = -std::sin(angle);
//	matrix[8] = std::sin(angle);
//	matrix[10] = std::cos(angle);
//}
//
//void rotateZ(float* matrix, float angle) {
//	matrix[0] = std::cos(angle);
//	matrix[1] = -std::sin(angle);
//	matrix[4] = std::sin(angle);
//	matrix[5] = std::cos(angle);
//}
//
//
//struct Vec3 {
//
//	float x, y, z;
//
//	void normalize() {
//		float length = std::sqrt(x * x + y * y + z * z);
//		x /= length; y /= length; z /= length;
//	}
//	static Vec3 crossMult(Vec3& vec1, Vec3& vec2) {
//		return Vec3{ vec1.y * vec2.z - vec2.y * vec1.z ,
//						vec2.x * vec1.z - vec1.x * vec2.z,
//						vec1.x * vec2.y - vec2.x * vec1.y };
//
//	}
//};
//
//Vec3 forwardCamera{ 0.0f, 0.0f, 0.0f };
//Vec3 camera{ 0.0f, 0.0f, 5.0f };
//Vec3 cameraUp{ 0.0f, 1.0f, 0.0f };
//bool firstMouse = true;
//bool cursorHidden = false;
//double mouseX, mouseY;
//float mouseSpeed = 0.02f;
//
//
//float pitchAngle = 0.0f;
//float yawAngle = -90.0f;
//
//
//void camCalc(float* viewMat, GLFWwindow* window) {
//	Vec3 rightVec = Vec3::crossMult(forwardCamera, cameraUp); rightVec.normalize();
//
//	Vec3 upVec = Vec3::crossMult(rightVec, forwardCamera); upVec.normalize();
//
//	static double startTime = glfwGetTime();
//	double currentTime = glfwGetTime();
//
//	float deltaTime = static_cast<float>(currentTime - startTime);
//
//	startTime = currentTime;
//	float velocity = 1.8f;
//
//	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
//		camera.x += forwardCamera.x * deltaTime * velocity;
//		camera.y += forwardCamera.y * deltaTime * velocity;
//		camera.z += forwardCamera.z * deltaTime * velocity;
//
//	}
//
//	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
//		camera.x -= forwardCamera.x * deltaTime * velocity;
//		camera.y -= forwardCamera.y * deltaTime * velocity;
//		camera.z -= forwardCamera.z * deltaTime * velocity;
//
//	}
//
//	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
//		camera.x += upVec.x * deltaTime * velocity;
//		camera.y += upVec.y * deltaTime * velocity;
//		camera.z += upVec.z * deltaTime * velocity;
//
//	}
//
//
//	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
//		camera.x -= upVec.x * deltaTime * velocity;
//		camera.y -= upVec.y * deltaTime * velocity;
//		camera.z -= upVec.z * deltaTime * velocity;
//
//	}
//
//
//	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
//		camera.x -= rightVec.x * deltaTime * velocity;
//		camera.y -= rightVec.y * deltaTime * velocity;
//		camera.z -= rightVec.z * deltaTime * velocity;
//
//	}
//
//
//	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
//		camera.x += rightVec.x * deltaTime * velocity;
//		camera.y += rightVec.y * deltaTime * velocity;
//		camera.z += rightVec.z * deltaTime * velocity;
//
//	}
//	viewMat[0] = rightVec.x;
//	viewMat[4] = rightVec.y;
//	viewMat[8] = rightVec.z;
//	viewMat[12] = -(rightVec.x * camera.x + rightVec.y * camera.y + rightVec.z * camera.z);
//
//
//
//	viewMat[1] = upVec.x;
//	viewMat[5] = upVec.y;
//	viewMat[9] = upVec.z;
//	viewMat[13] = -(upVec.x * camera.x + upVec.y * camera.y + upVec.z * camera.z);
//
//	viewMat[2] = -forwardCamera.x;
//	viewMat[6] = -forwardCamera.y;
//	viewMat[10] = -forwardCamera.z;
//	viewMat[14] = -(-forwardCamera.x * camera.x + -forwardCamera.y * camera.y + -forwardCamera.z * camera.z);
//
//
//}
//void calcForwardVector(float* viewMat, GLFWwindow* window) {
//	float pitchRad = pitchAngle * M_PI / 180.0f;
//	float yawRad = yawAngle * M_PI / 180.0f;
//
//	forwardCamera.x = std::cos(yawRad) * std::cos(pitchRad);
//	forwardCamera.y = std::sin(pitchRad);
//	forwardCamera.z = std::sin(yawRad) * std::cos(pitchRad);
//
//	forwardCamera.normalize();
//	camCalc(viewMat, window);
//
//
//}
//void enableCamera(float* viewMat, GLFWwindow* window) {
//	if (!cursorHidden) {
//		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
//		cursorHidden = true;
//	}
//
//	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
//		glfwSetWindowShouldClose(window, true);
//	}
//	double x, y;
//	glfwGetCursorPos(window, &x, &y);
//
//	if (firstMouse) {
//		mouseX = x;
//		mouseY = y;
//
//		firstMouse = false;
//	}
//
//	//MIGHT HAVE TO INVERT y - mouseY since it is opposite
//	//MIGHT HAVE TO INVERT y - mouseY since it is opposite
//	//MIGHT HAVE TO INVERT y - mouseY since it is opposite
//	float  mouseXDiff = static_cast<float>(x - mouseX);
//	float  mouseYDiff = static_cast<float>(mouseY - y);
//
//	mouseX = x;
//	mouseY = y;
//
//	mouseXDiff *= mouseSpeed;
//	mouseYDiff *= mouseSpeed;
//
//	yawAngle += mouseXDiff;
//	pitchAngle += mouseYDiff;
//
//	if (pitchAngle < -89.0f) {
//		pitchAngle = -89.0f;
//	}
//
//	if (pitchAngle > 89.0f) {
//		pitchAngle = 89.0f;
//	}
//	calcForwardVector(viewMat, window);
//}
//int main() {
//	if (!glfwInit()) {
//		std::cerr << "GLFW was not initialized\n";
//		return -1;
//
//	}
//	GLFWwindow* window = glfwCreateWindow(800, 800, "First program", NULL, NULL);
//	if (!window) {
//		std::cerr << "WINDOW WAS NOT INITIALIZED\n";
//		return -1;
//	}
//
//
//	glfwMakeContextCurrent(window);
//	glewExperimental = GL_TRUE;
//	if (glewInit() != GLEW_OK) {
//		std::cerr << "GLEW WAS NOT INITIALIZED\n";
//		return -1;
//	}
//
//
//	float buff[]{ 0.0f, 0.5f,
//				 -0.5f, -0.5f,
//				  0.5f, -0.5f };
//	unsigned int buffer;
//	//glGenBuffers(1, &buffer);
//	//glBindBuffer(GL_ARRAY_BUFFER, buffer);
//	//glBufferData(GL_ARRAY_BUFFER, sizeof(buff), buff, GL_STATIC_DRAW);
//
//	//glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
//	//glEnableVertexAttribArray(0);
//
//
//	//unsigned int program = getShaderProgram("BASIC.SHADER");
//
//
//
//	float cubeVertice[]{
//		//front face
//		0.5f,0.5f,0.5f,     1.0f, 1.0f, //top right			0
//		-0.5f,0.5f,0.5f,    0.0f, 1.0f, // top left			1
//		0.5f,-0.5f,0.5f,    1.0f, 0.0f, //bottom right		2
//		-0.5f,-0.5f,0.5f,   0.0f, 0.0f, //bottom left		3
//
//
//		//back face
//		-0.5f,0.5f,-0.5f,	0.0f, 1.0f, //top left			4
//		0.5f,0.5f,-0.5f,	1.0f, 1.0f, //top right			5
//		-0.5f,-0.5f,-0.5f,  0.0f, 0.0f, //bottom left		6
//		0.5f,-0.5f,-0.5f,   1.0f, 0.0f, //bottom right		7
//
//		//top face
//		0.5f,0.5f,0.5f,     0.0f, 0.0f, //bottom left		8
//		-0.5f,0.5f,0.5f,	0.0f, 1.0f, //top left			9
//		-0.5f,0.5f,-0.5f,   1.0f, 1.0f, //top right			10
//		0.5f,0.5f,-0.5f,	1.0f, 0.0f, //bottom right		11
//
//		//bottom face
//		0.5f,-0.5f,0.5f,	0.0f, 0.0f, //bottom left		12
//		-0.5f,-0.5f,0.5f,	0.0f, 1.0f, //top left			13
//		-0.5f,-0.5f,-0.5f,	1.0f, 1.0f, //top right			14
//		0.5f,-0.5f,-0.5f,	1.0f, 0.0f,	//bottom right		15					
//
//		//left face
//		-0.5f,0.5f,0.5f,	0.0f, 1.0f, //top left			16
//		-0.5f,0.5f,-0.5f,	1.0f, 1.0f, //top right			17
//		-0.5f,-0.5f,0.5f,	0.0f, 0.0f, //bottom left		18
//		-0.5f,-0.5f,-0.5f,	1.0f, 0.0f, //bottom right		19
//
//		//right face
//		0.5f,0.5f,0.5f,		0.0f, 1.0f,	//top left			20
//		0.5f,0.5f,-0.5f,	1.0f, 1.0f, //top right			21
//		0.5f,-0.5f,0.5f,	0.0f, 0.0f, //bottom left		22
//		0.5f,-0.5f,-0.5f,	1.0f, 0.0f	//bottom right		23
//
//
//
//	};
//
//	unsigned int indices[]{
//		//front face
//		1, 2, 3,
//		1, 2, 0,
//
//		//back face
//		4, 7, 6,
//		4, 7, 5,
//
//		//top face
//		9, 11, 8,
//		9, 11, 10,
//
//		//bottom face
//		13, 15, 12,
//		13, 15, 14,
//
//		//left face
//		16, 19, 18,
//		16, 19, 17,
//
//		//right face
//		20, 23, 22,
//		20, 23, 21
//
//
//	};
//
//	unsigned int indexBuff;
//	glGenBuffers(1, &indexBuff);
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuff);
//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
//	for (int i = 0; i <= 21; i += 3) {
//		std::cout << "(" << cubeVertice[i] << ", " << cubeVertice[i + 1] << ", " << cubeVertice[i + 2] << ")\n";
//
//	}
//
//
//	unsigned int program = getShaderProgram("BASIC.SHADER");
//	unsigned int buff2;
//	glGenBuffers(1, &buff2);
//	glBindBuffer(GL_ARRAY_BUFFER, buff2);
//	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertice), cubeVertice, GL_STATIC_DRAW);
//
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
//	glEnableVertexAttribArray(0);
//
//
//	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
//	glEnableVertexAttribArray(1);
//
//
//	float modelMat[16];
//	float projectionMat[16];
//	float viewMat[16];
//
//	setAsIdentityMat(modelMat);
//	setAsIdentityMat(projectionMat);
//	setAsIdentityMat(viewMat);
//
//	//SAVED PROJECTION MATRIX LEARNING IN MS PAINT
//	//SAVED PROJECTION MATRIX LEARNING IN MS PAINT
//	//
//
//	viewMat[14] = -5.00f;
//
//
//	float fov = 40.0f * (M_PI / 180.0f);  // Field of view in radians
//	float aspect = 800.0f / 800.0f;        // Width/Height
//	float near = 0.1f;
//	float far = 100.0f;
//
//	float f = 1.0f / tan(fov / 2.0f);
//
//	projectionMat[0] = f / aspect;  // X scaling
//	projectionMat[5] = f;            // Y scaling
//	projectionMat[10] = (far + near) / (near - far);
//	projectionMat[11] = -1.0f;      // Enables perspective divide
//	projectionMat[14] = (2.0f * far * near) / (near - far);
//	projectionMat[15] = 0.0f;
//
//
//
//
//	glUseProgram(program);
//
//	glUniformMatrix4fv(glGetUniformLocation(program, "model"), 1, GL_FALSE, modelMat);
//	glUniformMatrix4fv(glGetUniformLocation(program, "v iew"), 1, GL_FALSE, viewMat);
//	glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1, GL_FALSE, projectionMat);
//
//
//
//	stbi_set_flip_vertically_on_load(1);
//	std::string image = "C:/angry-bird.png";
//
//	int width, height, channels;
//
//	unsigned char* imagePtr = stbi_load(image.c_str(), &width, &height, &channels, 4);
//
//	if (imagePtr) {
//		std::cout << "width: " << width << ", " << " height: " << height << ", " << " channels: " << channels << "\n";
//	}
//	else {
//		std::cout << "COULDN'T FIND IMAGE\n";
//	}
//
//
//	unsigned int tex;
//	glGenTextures(1, &tex);
//	glBindTexture(GL_TEXTURE_2D, tex);
//
//	glActiveTexture(GL_TEXTURE0);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imagePtr);
//
//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//	glEnable(GL_BLEND);
//
//	glEnable(GL_DEPTH_TEST);
//
//	while (!glfwWindowShouldClose(window)) {
//		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
//		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//		glUseProgram(program);
//
//		float angle = -glfwGetTime() * 2.5f;
//		enableCamera(viewMat, window);
//		//rotateX(modelMat, angle);
//
//		glUniformMatrix4fv(glGetUniformLocation(program, "model"), 1, GL_FALSE, modelMat);
//		glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_FALSE, viewMat);
//		glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1, GL_FALSE, projectionMat);
//		glUniform1f(glGetUniformLocation(program, "texSamp"), 0);
//		glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (const void*)0);
//		glfwSwapBuffers(window);
//		glfwPollEvents();
//
//	}
//	return 0;
//}