// MinecraftClone.cpp - Fixed version with better visuals
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Block Types
enum BlockType {
    AIR = 0,
    GRASS = 1,
    DIRT = 2,
    STONE = 3,
    WOOD = 4,
    LEAVES = 5,
    WATER = 6,
    SAND = 7
};

// World Configuration
const int CHUNK_SIZE = 16;
const int WORLD_HEIGHT = 64;
const int RENDER_DISTANCE = 4;
const float BLOCK_SIZE = 1.0f;

// Camera/Player
struct Camera {
    glm::vec3 position = glm::vec3(0, 32, 0);
    glm::vec3 front = glm::vec3(0, 0, -1);
    glm::vec3 up = glm::vec3(0, 1, 0);
    glm::vec3 right = glm::vec3(1, 0, 0);
    float yaw = -90.0f;
    float pitch = 0.0f;
    float speed = 5.0f;
    float sensitivity = 0.1f;
    bool flying = true;
};

// Block vertex data with normals for proper lighting
float blockVertices[] = {
    // Positions          // Normals         // TexCoords
    // Front face (Z+)
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f,
     0.5f, -0.5f,  0.5f,  0.0f, 0.0f, 1.0f,  1.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  0.0f, 0.0f, 1.0f,  1.0f, 1.0f,
    -0.5f,  0.5f,  0.5f,  0.0f, 0.0f, 1.0f,  0.0f, 1.0f,

    // Back face (Z-)
     0.5f, -0.5f, -0.5f,  0.0f, 0.0f, -1.0f,  0.0f, 0.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 0.0f, -1.0f,  1.0f, 0.0f,
    -0.5f,  0.5f, -0.5f,  0.0f, 0.0f, -1.0f,  1.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  0.0f, 0.0f, -1.0f,  0.0f, 1.0f,

     // Left face (X-)
     -0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f,  0.0f, 0.0f,
     -0.5f, -0.5f,  0.5f, -1.0f, 0.0f, 0.0f,  1.0f, 0.0f,
     -0.5f,  0.5f,  0.5f, -1.0f, 0.0f, 0.0f,  1.0f, 1.0f,
     -0.5f,  0.5f, -0.5f, -1.0f, 0.0f, 0.0f,  0.0f, 1.0f,

     // Right face (X+)
      0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 0.0f,  0.0f, 0.0f,
      0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 0.0f,  1.0f, 0.0f,
      0.5f,  0.5f, -0.5f,  1.0f, 0.0f, 0.0f,  1.0f, 1.0f,
      0.5f,  0.5f,  0.5f,  1.0f, 0.0f, 0.0f,  0.0f, 1.0f,

      // Top face (Y+)
      -0.5f,  0.5f,  0.5f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f,
       0.5f,  0.5f,  0.5f,  0.0f, 1.0f, 0.0f,  1.0f, 0.0f,
       0.5f,  0.5f, -0.5f,  0.0f, 1.0f, 0.0f,  1.0f, 1.0f,
      -0.5f,  0.5f, -0.5f,  0.0f, 1.0f, 0.0f,  0.0f, 1.0f,

      // Bottom face (Y-)
      -0.5f, -0.5f, -0.5f,  0.0f, -1.0f, 0.0f,  0.0f, 0.0f,
       0.5f, -0.5f, -0.5f,  0.0f, -1.0f, 0.0f,  1.0f, 0.0f,
       0.5f, -0.5f,  0.5f,  0.0f, -1.0f, 0.0f,  1.0f, 1.0f,
      -0.5f, -0.5f,  0.5f,  0.0f, -1.0f, 0.0f,  0.0f, 1.0f
};


unsigned int blockIndices[] = {
    0,  1,  2,   0,  2,  3,   // front
    4,  5,  6,   4,  6,  7,   // back
    8,  9,  10,  8,  10, 11,  // left
    12, 13, 14,  12, 14, 15,  // right
    16, 17, 18,  16, 18, 19,  // top
    20, 21, 22,  20, 22, 23   // bottom
};

// Improved shaders with proper lighting
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in float aBlockType;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out float BlockType;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    BlockType = aBlockType;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in float BlockType;

uniform vec3 viewPos;

void main() {
    vec3 baseColor;
    
    // Minecraft-like color coding based on block type
    if (BlockType == 1.0) {        // GRASS
        baseColor = vec3(0.33, 0.7, 0.26);  // Rich green
    } else if (BlockType == 2.0) { // DIRT
        baseColor = vec3(0.55, 0.36, 0.23);  // Brown
    } else if (BlockType == 3.0) { // STONE
        baseColor = vec3(0.5, 0.5, 0.5);  // Gray
    } else if (BlockType == 4.0) { // WOOD
        baseColor = vec3(0.4, 0.27, 0.13);  // Dark brown
    } else if (BlockType == 5.0) { // LEAVES
        baseColor = vec3(0.13, 0.55, 0.13);  // Forest green
    } else if (BlockType == 6.0) { // WATER
        baseColor = vec3(0.2, 0.4, 0.8);  // Blue
    } else if (BlockType == 7.0) { // SAND
        baseColor = vec3(0.87, 0.78, 0.5);  // Sandy yellow
    } else {
        baseColor = vec3(0.8, 0.8, 0.8);  // Default
    }
    
    // Add subtle texture variation
    float pattern = sin(TexCoord.x * 32.0) * sin(TexCoord.y * 32.0) * 0.05;
    baseColor += pattern;
    
    // Directional lighting (sun from above and slightly to the side)
    vec3 lightDir = normalize(vec3(0.3, 1.0, 0.5));
    vec3 norm = normalize(Normal);
    
    // Diffuse lighting
    float diff = max(dot(norm, lightDir), 0.0);
    
    // Ambient lighting (stronger to prevent too dark areas)
    float ambient = 0.5;
    
    // Face-based shading for Minecraft look
    float faceShading = 1.0;
    if (abs(norm.y) > 0.9) {
        // Top/bottom faces
        faceShading = (norm.y > 0.0) ? 1.0 : 0.5;  // Top bright, bottom dark
    } else if (abs(norm.x) > 0.9) {
        // Left/right faces
        faceShading = 0.8;
    } else {
        // Front/back faces
        faceShading = 0.6;
    }
    
    // Combine lighting
    float lighting = ambient + diff * 0.5;
    lighting *= faceShading;
    
    // Apply lighting to color
    vec3 finalColor = baseColor * lighting;
    
    // Add slight fog for depth
    float fogStart = 40.0;
    float fogEnd = 80.0;
    float fogDistance = length(FragPos - viewPos);
    float fogFactor = clamp((fogEnd - fogDistance) / (fogEnd - fogStart), 0.0, 1.0);
    vec3 fogColor = vec3(0.5, 0.75, 1.0); // Sky blue
    
    finalColor = mix(fogColor, finalColor, fogFactor);
    
    FragColor = vec4(finalColor, 1.0);
}
)";

// Perlin noise-like function for better terrain
float noise(int x, int z, int seed) {
    int n = x + z * 57 + seed * 131;
    n = (n << 13) ^ n;
    return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
}

float smoothNoise(float x, float z, int seed) {
    int intX = (int)x;
    int intZ = (int)z;
    float fracX = x - intX;
    float fracZ = z - intZ;

    float v1 = noise(intX, intZ, seed);
    float v2 = noise(intX + 1, intZ, seed);
    float v3 = noise(intX, intZ + 1, seed);
    float v4 = noise(intX + 1, intZ + 1, seed);

    float i1 = v1 * (1 - fracX) + v2 * fracX;
    float i2 = v3 * (1 - fracX) + v4 * fracX;

    return i1 * (1 - fracZ) + i2 * fracZ;
}

float interpolatedNoise(float x, float z, int seed) {
    float total = 0;
    float frequency = 0.05f;
    float amplitude = 1.0f;

    for (int i = 0; i < 4; i++) {
        total += smoothNoise(x * frequency, z * frequency, seed + i) * amplitude;
        frequency *= 2;
        amplitude *= 0.5;
    }

    return total;
}

// World/Chunk Management
class Chunk {
public:
    BlockType blocks[CHUNK_SIZE][WORLD_HEIGHT][CHUNK_SIZE];
    int chunkX, chunkZ;
    std::vector<float> vertices; 
    std::vector<unsigned int> indices;
    unsigned int VAO, VBO, EBO;
    bool needsUpdate = true;

    Chunk(int x, int z) : chunkX(x), chunkZ(z) {
        generateTerrain();
        setupMesh();
    }

    void generateTerrain() {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int z = 0; z < CHUNK_SIZE; z++) {

                // Improved terrain generation with noise
                int worldX = chunkX * CHUNK_SIZE + x;
                int worldZ = chunkZ * CHUNK_SIZE + z;


                float heightValue = interpolatedNoise(worldX, worldZ, 12345);
                int baseHeight = 25 + (int)(heightValue * 15);
                baseHeight = std::max(5, std::min(WORLD_HEIGHT - 10, baseHeight));
                //std::cout << "baseHeight: " << baseHeight << "\n";


                for (int y = 0; y < WORLD_HEIGHT; y++) {
                    if (y == 0) {
                        blocks[x][y][z] = STONE; // Bedrock
                    }
                    else if (y < baseHeight - 4) {
                        blocks[x][y][z] = STONE;
                    }
                    else if (y < baseHeight - 1) {
                        blocks[x][y][z] = DIRT;
                    }
                    else if (y == baseHeight - 1) {
                        blocks[x][y][z] = GRASS;
                    }
                    else {
                        blocks[x][y][z] = AIR;
                    }
                }

                // Better tree generation
                 //NOTE TO SELF: To explain what is happening here for blocks[x][baseHeight - 1][z] == GRASS, this condition is always true because in the previous for loop "for (int y = 0; y < WORLD_HEIGHT; y++) " we always use y = baseHeight - 1 for grass, and we are using the same x and z from the outer loop "for (int x = 0; x < CHUNK_SIZE; x++) {" '  for (int z = 0; z < CHUNK_SIZE; z++) {" in the previous for loop and this condition, thus since x, and z, are the same and y(baseHeight -1 ) in addition to this was used for grass, and in the if statement we use the same x and z, and same y, it always has to be true
                //NOTE TO SELF: The reason we need this condition "x > 2 && x < CHUNK_SIZE - 3 && z > 2 && z < CHUNK_SIZE - 3" is because of   if (blocks[x + dx][leafY][z + dz] == AIR), if x was less than 2, say 1, the minimum of dx is -2, so we would be doing 1 + -2, which is -1, and we cant have negative index, and we need x and z to not be greater than CHUNK_SIZE - 3 because if it is like for example if it was 14(note CHUNK_SIZE - 3 is 16 -3 = 13), and we know the maximum number for dx and dz is 2, we would do 14 + 2 which is 16 and note the blocks array is   BlockType blocks[CHUNK_SIZE][WORLD_HEIGHT][CHUNK_SIZE];, so we could be doing something like blocks[16][leafY][16] but we cannot access index 16 since the size is 16(valid indices includes 0 - 15), but we don't need to make thisi condition strict because    x >= 2 && x <= CHUNK_SIZE - 3 && z >= 2 && z <= CHUNK_SIZE - 3 also works if x and z are 2 we get 0 with like dx+ x, and 13 or CHUNK_SIZE -3 + dx maximum which is 2, is 15 which is valid
                if (blocks[x][baseHeight - 1][z] == GRASS &&
                    x > 2 && x < CHUNK_SIZE - 3 && z > 2 && z < CHUNK_SIZE - 3) {

                    // Use noise for tree placement
                    float treeNoise = noise(worldX, worldZ, 54321);
                    if (treeNoise > 0.85) {
                        int treeHeight = 4 + (gen() % 2);

                        // Tree trunk
                        for (int y = baseHeight; y < baseHeight + treeHeight; y++) {
                            if (y < WORLD_HEIGHT) blocks[x][y][z] = WOOD;
                        }

                        // Tree leaves (larger, more natural shape)
                        for (int dx = -2; dx <= 2; dx++) {
                            for (int dz = -2; dz <= 2; dz++) {
                                for (int dy = 0; dy < 3; dy++) {
                                    int leafY = baseHeight + treeHeight - 2 + dy;

                                    // Create rounded leaf shape
                                    if (abs(dx) + abs(dz) <= 2 + (dy == 1 ? 1 : 0)) {
                                        if (leafY < WORLD_HEIGHT &&
                                            x + dx >= 0 && x + dx < CHUNK_SIZE &&
                                            z + dz >= 0 && z + dz < CHUNK_SIZE) {
                                            if (blocks[x + dx][leafY][z + dz] == AIR) {
                                                blocks[x + dx][leafY][z + dz] = LEAVES;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Top of tree
                        if (baseHeight + treeHeight < WORLD_HEIGHT) {
                            blocks[x][baseHeight + treeHeight][z] = LEAVES;
                        }
                    }
                }
            }
        }

        //for (int y = 0; y < WORLD_HEIGHT; ++y) {
        //    std::string blockType;
        //    if (blocks[0][y][0] == STONE) {
        //        blockType = "STONE";
        //    }
        //    if (blocks[0][y][0] == DIRT) {
        //        blockType = "DIRT";
        //    }
        //    if (blocks[0][y][0] == GRASS) {
        //        blockType = "GRASS";
        //    }
        //    if (blocks[0][y][0] == WOOD) {
        //        blockType = "WOOD";
        //    }
        //    if (blocks[0][y][0] == AIR) {
        //        blockType = "AIR";
        //    }
        //    std::cout << "(" << 0 << ", " << y << ", " << 0 << ")   BLOCK: " << blockType << "\n";
        //}
    }
    void setupMesh() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
    }

    bool isBlockSolid(int x, int y, int z) {
        if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= WORLD_HEIGHT || z < 0 || z >= CHUNK_SIZE) {
            return false;
        }
        return blocks[x][y][z] != AIR;
    }

    void generateMesh() {
        vertices.clear();
        indices.clear();

        unsigned int indexOffset = 0;

        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < WORLD_HEIGHT; y++) {
                for (int z = 0; z < CHUNK_SIZE; z++) {
                    if (blocks[x][y][z] == AIR) continue;

                    glm::vec3 pos(chunkX * CHUNK_SIZE + x, y, chunkZ * CHUNK_SIZE + z);
                    BlockType currentBlock = blocks[x][y][z];

                    // Check each face for visibility
                    //CONFUSION HERE: WE ARE NEGATING EACH VALUE IN ARRAY WITH " ! " 
                    //NOTE TO SELF: if we only have one iteration in generateTerrain function   for (int x = 0; x < 1; x++) {   for (int y = 0; y < 1; y++) { other unassigned indices of     BlockType blocks[CHUNK_SIZE][WORLD_HEIGHT][CHUNK_SIZE]; is 0 or AIR but can check to make sure, so at (0, 0, 0), faces checks like (0, + 1, 0) would be those unassigned indices which are AIR and if we check return blocks[x][y][z] != AIR; in the function we get false and we negate false in the array "bool faces[6] " for it to be true to render the face
                    //NOTE TO SELF: RENDERS EXTERIOR OF CUBE NOT FACES INSIDE CHUNK, VIEW RENDERING_FACE image saved on pc for reference
                    bool faces[6] = {
          

                         
                        !isBlockSolid(x, y, z + 1), // front
                        !isBlockSolid(x, y, z - 1), // Back
                        !isBlockSolid(x - 1, y, z), // left
                        !isBlockSolid(x + 1, y, z), // right
                        !isBlockSolid(x, y + 1, z), // top
                        !isBlockSolid(x, y - 1, z)  // bottom
                    };

                    for (int face = 0; face < 6; face++) {
                        //if (face == 4) std::cout << "top face rendered: " << faces[face]  << ",     x: " << x << ", y: " << y << ", z: " << z << "\n";
                        if (!faces[face]) continue;

                        // Add vertices for this face (position, normal, texcoord, blocktype)
                        for (int i = 0; i < 4; i++) {
                            int vertexIndex = face * 4 + i;
                            // Position
                            vertices.push_back(blockVertices[vertexIndex * 8 + 0] + pos.x );
                            vertices.push_back(blockVertices[vertexIndex * 8 + 1] + pos.y);
                            vertices.push_back(blockVertices[vertexIndex * 8 + 2] + pos.z);
                            // Normal
                            vertices.push_back(blockVertices[vertexIndex * 8 + 3]);
                            vertices.push_back(blockVertices[vertexIndex * 8 + 4]);
                            vertices.push_back(blockVertices[vertexIndex * 8 + 5]);
                            // TexCoord
                            vertices.push_back(blockVertices[vertexIndex * 8 + 6]);
                            vertices.push_back(blockVertices[vertexIndex * 8 + 7]);
                            // Block type
                            vertices.push_back((float)currentBlock);
                        }

                        // Add indices
                        unsigned int faceIndices[] = { 0, 1, 2, 0, 2, 3 };
                        for (int i = 0; i < 6; i++) {
                            indices.push_back(indexOffset + faceIndices[i]);
                        }
                        indexOffset += 4;
                    }
                }
            }
        }

        // Upload to GPU
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        // Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // Texture coordinate attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);

        // Block type attribute
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(8 * sizeof(float)));
        glEnableVertexAttribArray(3);

        needsUpdate = false;
    }

    void render() {
        if (needsUpdate) {
            generateMesh();
        }

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    }

    ~Chunk() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
    }
};

// Game World
class World {
public:
    std::map<std::pair<int, int>, Chunk*> chunks;

    Chunk* getChunk(int chunkX, int chunkZ) {
        auto key = std::make_pair(chunkX, chunkZ);
        if (chunks.find(key) == chunks.end()) {
            chunks[key] = new Chunk(chunkX, chunkZ);
        }
        return chunks[key];
    }

    BlockType getBlock(int x, int y, int z) {
        if (y < 0 || y >= WORLD_HEIGHT) return AIR;

        int chunkX = x / CHUNK_SIZE;
        int chunkZ = z / CHUNK_SIZE;
        int localX = x % CHUNK_SIZE;
        int localZ = z % CHUNK_SIZE;

        if (localX < 0) { localX += CHUNK_SIZE; chunkX--; }
        if (localZ < 0) { localZ += CHUNK_SIZE; chunkZ--; }

        Chunk* chunk = getChunk(chunkX, chunkZ);
        return chunk->blocks[localX][y][localZ];
    }

    void setBlock(int x, int y, int z, BlockType block) {
        if (y < 0 || y >= WORLD_HEIGHT) return;

        int chunkX = x / CHUNK_SIZE;
        int chunkZ = z / CHUNK_SIZE;
        int localX = x % CHUNK_SIZE;
        int localZ = z % CHUNK_SIZE;

        if (localX < 0) { localX += CHUNK_SIZE; chunkX--; }
        if (localZ < 0) { localZ += CHUNK_SIZE; chunkZ--; }

        Chunk* chunk = getChunk(chunkX, chunkZ);
        chunk->blocks[localX][y][localZ] = block;
        chunk->needsUpdate = true;

        // Update neighboring chunks if on edge
        if (localX == 0) getChunk(chunkX - 1, chunkZ)->needsUpdate = true;
        if (localX == CHUNK_SIZE - 1) getChunk(chunkX + 1, chunkZ)->needsUpdate = true;
        if (localZ == 0) getChunk(chunkX, chunkZ - 1)->needsUpdate = true;
        if (localZ == CHUNK_SIZE - 1) getChunk(chunkX, chunkZ + 1)->needsUpdate = true;
    }

    void renderAroundPlayer(glm::vec3 playerPos) {
        int playerChunkX = (int)floor(playerPos.x / CHUNK_SIZE);
        int playerChunkZ = (int)floor(playerPos.z / CHUNK_SIZE);

       /* std::cout << "playerPos.x: " << playerPos.x << "\n";
        std::cout << "playerChunkX: " << playerChunkX << "\n";*/
       /* std::cout << "playerPos.x: " << playerPos.x << ", playerPos.z: " << playerPos.z << '\n';
        std::cout << "playerChunkX: " << playerChunkX << ", playerChunkZ: " << playerChunkZ << '\n';*/

        for (int x = playerChunkX - RENDER_DISTANCE; x <= playerChunkX + RENDER_DISTANCE; x++) {
            for (int z = playerChunkZ - RENDER_DISTANCE; z <= playerChunkZ + RENDER_DISTANCE; z++) {
               // std::cout << "x: " << x << "\n";
                Chunk* chunk = getChunk(x, z);
                chunk->render();
            }
        }
    }

    ~World() {
        for (auto& pair : chunks) {
            delete pair.second;
        }
    }
};

// Shader Compilation
unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return shader;
}

unsigned int createShaderProgram() {
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

// Input handling
Camera camera;
World world;
bool keys[1024];
bool firstMouse = true;
float lastX = 400, lastY = 300;
BlockType selectedBlock = STONE;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS) keys[key] = true;
        else if (action == GLFW_RELEASE) keys[key] = false;
    }

    // Block selection
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_1) selectedBlock = GRASS;
        if (key == GLFW_KEY_2) selectedBlock = DIRT;
        if (key == GLFW_KEY_3) selectedBlock = STONE;
        if (key == GLFW_KEY_4) selectedBlock = WOOD;
        if (key == GLFW_KEY_5) selectedBlock = LEAVES;
        if (key == GLFW_KEY_6) selectedBlock = WATER;
        if (key == GLFW_KEY_7) selectedBlock = SAND;
    }
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    xoffset *= camera.sensitivity;
    yoffset *= camera.sensitivity;

    camera.yaw += xoffset;
    camera.pitch += yoffset;

    if (camera.pitch > 89.0f) camera.pitch = 89.0f;
    if (camera.pitch < -89.0f) camera.pitch = -89.0f;

    glm::vec3 direction;
    direction.x = cos(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
    direction.y = sin(glm::radians(camera.pitch));
    direction.z = sin(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
    camera.front = glm::normalize(direction);
    camera.right = glm::normalize(glm::cross(camera.front, glm::vec3(0, 1, 0)));
    camera.up = glm::normalize(glm::cross(camera.right, camera.front));
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        glm::vec3 ray = camera.front;
        glm::vec3 pos = camera.position;

        for (float t = 0; t < 10.0f; t += 0.1f) {
            glm::vec3 testPos = pos + ray * t;
            int x = (int)floor(testPos.x);
            int y = (int)floor(testPos.y);
            int z = (int)floor(testPos.z);

            if (world.getBlock(x, y, z) != AIR) {
                if (button == GLFW_MOUSE_BUTTON_LEFT) {
                    world.setBlock(x, y, z, AIR);
                }
                else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                    glm::vec3 placePos = pos + ray * (t - 0.2f);
                    int px = (int)floor(placePos.x);
                    int py = (int)floor(placePos.y);
                    int pz = (int)floor(placePos.z);
                    if (world.getBlock(px, py, pz) == AIR) {
                        world.setBlock(px, py, pz, selectedBlock);
                    }
                }
                break;
            }
        }
    }
}

void processInput(GLFWwindow* window, float deltaTime) {
    float velocity = camera.speed * deltaTime;

    if (keys[GLFW_KEY_W]) camera.position += camera.front * velocity;
    if (keys[GLFW_KEY_S]) camera.position -= camera.front * velocity;
    if (keys[GLFW_KEY_A]) camera.position -= camera.right * velocity;
    if (keys[GLFW_KEY_D]) camera.position += camera.right * velocity;
    if (keys[GLFW_KEY_SPACE]) camera.position += camera.up * velocity;
    if (keys[GLFW_KEY_LEFT_SHIFT]) camera.position -= camera.up * velocity;
}

// Main function
int main() {
   
    std::cout << "\n";
    // Initialize GLFW
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1200, 800, "Minecraft Clone - Enhanced", NULL, NULL);
    if (!window) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // OpenGL settings
    glEnable(GL_DEPTH_TEST);
   /* glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);*/

    // Create shader program
    unsigned int shaderProgram = createShaderProgram();

    // Game loop timing
    float deltaTime = 0.0f;
    float lastFrame = 0.0f;

    int counter = 0;
    // Generate initial chunks around spawn
    for (int x = -2; x <= 2; x++) {
        for (int z = -2; z <= 2; z++) {
           //world.getChunk(x, z)->render();
        }
    }
    std::cout << "Counter: " << counter << "\n";

    std::cout << "=== MINECRAFT CLONE - ENHANCED ===" << std::endl;
    std::cout << "WASD: Move" << std::endl;
    std::cout << "Mouse: Look around" << std::endl;
    std::cout << "Space: Fly up" << std::endl;
    std::cout << "Shift: Fly down" << std::endl;
    std::cout << "Left Click: Break block" << std::endl;
    std::cout << "Right Click: Place block" << std::endl;
    std::cout << "1-7: Select block type" << std::endl;
    std::cout << "ESC: Exit" << std::endl;

    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window, deltaTime);

        glClearColor(0.5f, 0.75f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // Set up matrices
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1200.0f / 800.0f, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(camera.position, camera.position + camera.front, camera.up);
        glm::mat4 model = glm::mat4(1.0f);

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(camera.position));

        // Render world
        world.renderAroundPlayer(camera.position);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    return 0;
}