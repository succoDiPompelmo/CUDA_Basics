#include "gpurasteriser.cuh"
#include "utilities/OBJLoader.hpp"
#include <vector>
#include <iomanip>
#include <chrono>
#include <limits>
#include <iostream>
#include <algorithm>
#include "cuda_runtime.h"
#include "utilities/cuda_error_helper.hpp"
#include "utilities/geometry.hpp"

struct workItemGPU {
    float scale;
    float3 distanceOffset;

    workItemGPU(float& scale_, float3& distanceOffset_) : scale(scale_), distanceOffset(distanceOffset_) {}
    workItemGPU() : scale(1), distanceOffset(make_float3(0, 0, 0)) {}
};

const std::vector<globalLight> lightSources = { {{0.3f, 0.5f, 1.0f}, {1.0f, 1.0f, 1.0f}} };








void runVertexShader( float4 &vertex,
                      float3 positionOffset,
                      float scale,
					  unsigned int const width,
					  unsigned int const height,
				  	  float const rotationAngle = 0)
{
	float const pi = 3.1415926f;
	// The matrices defined below are the ones used to transform the vertices and normals.

	// This projection matrix assumes a 16:9 aspect ratio, and an field of view (FOV) of 90 degrees.
	mat4x4 const projectionMatrix(
		0.347270,   0, 			0, 		0,
		0,	  		0.617370, 	0,		0,
		0,	  		0,			-1, 	-0.2f,
		0,	  		0,			-1,		0);

	mat4x4 translationMatrix(
		1,			0,			0,			0 + positionOffset.x /*X*/,
		0,			1,			0,			0 + positionOffset.y /*Y*/,
		0,			0,			1,			-10 + positionOffset.z /*Z*/,
		0,			0,			0,			1);

	mat4x4 scaleMatrix(
		scale/*X*/,	0,			0,				0,
		0, 			scale/*Y*/, 0,				0,
		0, 			0,			scale/*Z*/, 	0,
		0, 			0,			0,				1);

	mat4x4 const rotationMatrixX(
		1,			0,				0, 				0,
		0, 			std::cos(0), 	-std::sin(0),	0,
		0, 			std::sin(0),	std::cos(0), 	0,
		0, 			0,				0,				1);

	float const rotationAngleRad = (pi / 4.0f) + (rotationAngle / (180.0f/pi));

	mat4x4 const rotationMatrixY(
		std::cos(rotationAngleRad),		0,			std::sin(rotationAngleRad), 	0,
		0, 								1, 			0,								0,
		-std::sin(rotationAngleRad), 	0,			std::cos(rotationAngleRad), 	0,
		0, 								0,			0,								1);

	mat4x4 const rotationMatrixZ(
		std::cos(pi),	-std::sin(pi),	0,			0,
		std::sin(pi), 	std::cos(pi), 	0,			0,
		0,				0,				1,			0,
		0, 				0,				0,			1);

	mat4x4 const MVP =
		projectionMatrix * translationMatrix * rotationMatrixX * rotationMatrixY * rotationMatrixZ * scaleMatrix;

		float4 transformed = (MVP * vertex);

    vertex.x = transformed.x / transformed.w;
    vertex.y = transformed.y / transformed.w;
    vertex.z = transformed.z / transformed.w;
    vertex.w = 1.0;

    vertex.x = (vertex.x + 0.5f) * (float) width;
    vertex.y = (vertex.y + 0.5f) * (float) height;
}


void runFragmentShader( unsigned char* frameBuffer,
						unsigned int const baseIndex,
						GPUMesh &mesh,
						unsigned int triangleIndex,
						float3 const &weights)
{
	float3 normal = computeInterpolatedNormal(
            mesh.normals[3 * triangleIndex + 0],
            mesh.normals[3 * triangleIndex + 1],
            mesh.normals[3 * triangleIndex + 2],
			weights);

    float3 colour = make_float3(0.0f, 0.0f, 0.0f);

	for (globalLight const &l : lightSources) {
		float lightNormalDotProduct =
			normal.x * l.direction.x + normal.y * l.direction.y + normal.z * l.direction.z;

		float3 diffuseReflectionColour;
		diffuseReflectionColour.x = mesh.objectDiffuseColour.x * l.colour.x;
		diffuseReflectionColour.y = mesh.objectDiffuseColour.y * l.colour.y;
		diffuseReflectionColour.z = mesh.objectDiffuseColour.z * l.colour.z;

		colour.x += diffuseReflectionColour.x * lightNormalDotProduct;
		colour.y += diffuseReflectionColour.y * lightNormalDotProduct;
		colour.z += diffuseReflectionColour.z * lightNormalDotProduct;
	}

    colour.x = std::min(std::max(colour.x, 0.0f), 1.0f);
    colour.y = std::min(std::max(colour.y, 0.0f), 1.0f);
    colour.z = std::min(std::max(colour.z, 0.0f), 1.0f);

    frameBuffer[4 * baseIndex + 0] = colour.x * 255.0f;
    frameBuffer[4 * baseIndex + 1] = colour.y * 255.0f;
    frameBuffer[4 * baseIndex + 2] = colour.z * 255.0f;
    frameBuffer[4 * baseIndex + 3] = 255;

}

/**
 * The main procedure which rasterises all triangles on the framebuffer
 * @param transformedMesh         Transformed mesh object
 * @param frameBuffer             frame buffer for the rendered image
 * @param depthBuffer             depth buffer for every pixel on the image
 * @param width                   width of the image
 * @param height                  height of the image
 */
void rasteriseTriangle( float4 &v0, float4 &v1, float4 &v2,
                        GPUMesh &mesh,
                        unsigned int triangleIndex,
                        unsigned char* frameBuffer,
                        float* depthBuffer,
                        unsigned int const width,
                        unsigned int const height ) {

    // Compute the bounding box of the triangle.
    // Pixels that are intersecting with the triangle can only lie in this rectangle
	unsigned int minx = unsigned(std::floor(std::min(std::min(v0.x, v1.x), v2.x)));
	unsigned int maxx = unsigned(std::ceil(std::max(std::max(v0.x, v1.x), v2.x)));
	unsigned int miny = unsigned(std::floor(std::min(std::min(v0.y, v1.y), v2.y)));
	unsigned int maxy = unsigned(std::ceil(std::max(std::max(v0.y, v1.y), v2.y)));

	// Make sure the screen coordinates stay inside the window
    // This ensures parts of the triangle that are outside the
    // view of the camera are not drawn.
	minx = std::max(minx, (unsigned int) 0);
	maxx = std::min(maxx, width);
	miny = std::max(miny, (unsigned int) 0);
	maxy = std::min(maxy, height);

	// We iterate over each pixel in the triangle's bounding box
	for (unsigned int x = minx; x < maxx; x++) {
		for (unsigned int y = miny; y < maxy; y++) {
			float u, v, w;
			// For each point in the bounding box, determine whether that point lies inside the triangle
			if (inPointInTriangle(v0, v1, v2, x, y, u, v, w)) {
				// If it does, compute the distance between that point on the triangle and the screen
				float pixelDepth = computeDepth(v0, v1, v2, make_float3(u, v, w));
				// If the point is closer than any point we have seen thus far, render it.
				// Otherwise it is hidden behind another object, and we can throw it away
				// Because it will be invisible anyway.
                if (pixelDepth >= -1 && pixelDepth <= 1 && pixelDepth < depthBuffer[y * width + x]) {
				    // If it is, we update the depth buffer to the new depth.
					depthBuffer[y * width + x] = pixelDepth;
					runFragmentShader(frameBuffer, x + (width * y), mesh, triangleIndex, make_float3(u, v, w));
				}
			}
		}
	}
}


void renderMeshes(
        unsigned long totalItemsToRender,
        workItemGPU* workQueue,
        GPUMesh* meshes,
        unsigned int meshCount,
        unsigned int width,
        unsigned int height,
        unsigned char* frameBuffer,
        float* depthBuffer
) {

    for(unsigned int item = 0; item < totalItemsToRender; item++) {
        workItemGPU objectToRender = workQueue[item];
        for (unsigned int meshIndex = 0; meshIndex < meshCount; meshIndex++) {
            for(unsigned int triangleIndex = 0; triangleIndex < meshes[meshIndex].vertexCount / 3; triangleIndex++) {
                float4 v0 = meshes[meshIndex].vertices[triangleIndex * 3 + 0];
                float4 v1 = meshes[meshIndex].vertices[triangleIndex * 3 + 1];
                float4 v2 = meshes[meshIndex].vertices[triangleIndex * 3 + 2];

                runVertexShader(v0, objectToRender.distanceOffset, objectToRender.scale, width, height);
                runVertexShader(v1, objectToRender.distanceOffset, objectToRender.scale, width, height);
                runVertexShader(v2, objectToRender.distanceOffset, objectToRender.scale, width, height);

                rasteriseTriangle(v0, v1, v2, meshes[meshIndex], triangleIndex, frameBuffer, depthBuffer, width, height);
            }
        }
    }
}



void fillWorkQueue(
        workItemGPU* workQueue,
        float largestBoundingBoxSide,
        int depthLimit,
        unsigned long* nextIndexInQueue,
        float scale = 1.0,
        float3 distanceOffset = {0, 0, 0}) {

    // Queue a work item at the current scale and location
    workQueue[*nextIndexInQueue] = {scale, distanceOffset};
    (*nextIndexInQueue)++;

    // Check whether we've reached the recursive depth of the fractal we want to reach
    depthLimit--;
    if(depthLimit == 0) {
        return;
    }

    // Now we recursively draw the meshes in a smaller size
    for(int offsetX = -1; offsetX <= 1; offsetX++) {
        for(int offsetY = -1; offsetY <= 1; offsetY++) {
            for(int offsetZ = -1; offsetZ <= 1; offsetZ++) {
                float3 offset = make_float3(offsetX,offsetY,offsetZ);
                // We draw the new objects in a grid around the "main" one.
                // We thus skip the location of the object itself.
                if(offsetX == 0 && offsetY == 0 && offsetZ == 0) {
                    continue;
                }

                float smallerScale = scale / 3.0f;
                float3 displacedOffset = make_float3(
                        distanceOffset.x + offset.x * (largestBoundingBoxSide / 2.0f) * scale,
                        distanceOffset.y + offset.y * (largestBoundingBoxSide / 2.0f) * scale,
                        distanceOffset.z + offset.z * (largestBoundingBoxSide / 2.0f) * scale
                );

                fillWorkQueue(workQueue, largestBoundingBoxSide, depthLimit, nextIndexInQueue, smallerScale, displacedOffset);
            }
        }
    }

}

// Kernel definition
__global__ void frameBufferInitialisation(unsigned char *GPUframeBuffer)
{
  int index = (blockIdx.x) * 1024 + threadIdx.x * 4 + threadIdx.y;
  if (threadIdx.y == 3) GPUframeBuffer[index] = 255;
  else GPUframeBuffer[index] = 0;
}

__global__ void depthBufferInitialisation(float *GPUdepthBuffer)
{
  GPUdepthBuffer[(blockIdx.x * 5 + blockIdx.y) * 1024 + threadIdx.x] = 0;
}

// This function kicks off the rasterisation process.
std::vector<unsigned char> rasteriseGPU(std::string inputFile, unsigned int width, unsigned int height, unsigned int depthLimit) {
    std::cout << "Rendering an image on the GPU.." << std::endl;
    std::cout << "Loading '" << inputFile << "' file... " << std::endl;

    std::vector<GPUMesh> meshes = loadWavefrontGPU(inputFile, false);

    // CUDA INITIALISATION

    int nDevices;
    checkCudaErrors(cudaGetDeviceCount(&nDevices));
    std::cout << "\n" << "--- Devices ---" << '\n';

    for (int i = 0; i < nDevices; i++)
    {
      cudaDeviceProp prop;
      checkCudaErrors(cudaGetDeviceProperties(&prop, i));
      std::cout << "Device Number:  " << i << '\n';
      std::cout << "Device Name:  " << prop.name << '\n';
      std::cout << "Memory Clock Rate:  " << prop.memoryClockRate << '\n';
      std::cout << "Memory Bus Width (bits):  " << prop.memoryBusWidth << '\n';
      std::cout << "Peak Memory Bandwidth (GB/s):  " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth/8)/1.0e6 << '\n';
    }

    std::cout << "--- END Devices---" << '\n' << "\n";

    // Set device for GPU computation
    checkCudaErrors(cudaSetDevice(0));

    // Depth buffer allocation on the GPU
    float *GPUdepthBuffer;
    checkCudaErrors(cudaMalloc((void **)&GPUdepthBuffer, sizeof(float) * width * height));
    // Kernel - Depth buffer initialisation
    dim3 numBlocksD(5, 405);
    dim3 threadsPerBlockD(1024, 1);
    depthBufferInitialisation<<<numBlocksD, threadsPerBlockD>>>(GPUdepthBuffer);
    // Wait for the kernel to finish the computation
    checkCudaErrors(cudaDeviceSynchronize());

    // Frame buffer allocation on the GPU
    unsigned char *GPUframeBuffer;
    checkCudaErrors(cudaMalloc((void **)&GPUframeBuffer, sizeof(unsigned char) * width * height * 4));
    // Kernel - Frame buffer initialisation
    dim3 numBlocksF(8100, 1);
    dim3 threadsPerBlockF(256, 4);
    frameBufferInitialisation<<<numBlocksF, threadsPerBlockF>>>(GPUframeBuffer);
    // Wait for the kernel to finish the computation
    checkCudaErrors(cudaDeviceSynchronize());

    unsigned char* frameBufferTest = new unsigned char[width * height * 4];

    // Test if the frame buffer on the GPU contain the right value
    checkCudaErrors(cudaMemcpy(frameBufferTest, GPUframeBuffer, sizeof(unsigned char) * width * height * 4, cudaMemcpyDeviceToHost));
    std::cout << "\n" << "FRAME BUFFER TEST : ";
    unsigned int counterF = 0;
    unsigned int counterF0 = 0;
    for (unsigned int i = 0; i < width * height * 4; i++)
    {
      if ((int) frameBufferTest[i] == 255) counterF++;
      else counterF0++;
    }
    if (counterF == width * height && counterF0 == width * height * 3) std::cout << "PASSED" << "\n" << '\n';

    // Test if the depth buffer on the GPU contain the right value
    float* depthBufferTest = new float[width * height];
    checkCudaErrors(cudaMemcpy(depthBufferTest, GPUdepthBuffer, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
    std::cout << "\n" << "DEPTH BUFFER TEST : ";
    unsigned int counterZ = 0;
    for (unsigned int i = 0; i < width * height; i++)
    {
      if ((int) depthBufferTest[i] == 0) counterZ++;
    }
    if (counterZ == width * height) std::cout << "PASSED" << '\n' << "\n";

    // We first need to allocate some buffers.
    // The framebuffer contains the image being rendered.
    unsigned char* frameBuffer = new unsigned char[width * height * 4];
    // The depth buffer is used to make sure that objects closer to the camera occlude/obscure objects that are behind it
    for (unsigned int i = 0; i < (4 * width * height); i+=4) {
		frameBuffer[i + 0] = 0;
		frameBuffer[i + 1] = 0;
		frameBuffer[i + 2] = 0;
		frameBuffer[i + 3] = 255;
	}

	float* depthBuffer = new float[width * height];
	for(unsigned int i = 0; i < width * height; i++) {
    	depthBuffer[i] = 1;
    }

    float3 boundingBoxMin = make_float3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    float3 boundingBoxMax = make_float3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());

    std::cout << "Rendering image... " << std::endl;

    for(unsigned int i = 0; i < meshes.size(); i++) {
        for(unsigned int vertex = 0; vertex < meshes.at(i).vertexCount; vertex++) {
            boundingBoxMin.x = std::min(boundingBoxMin.x, meshes.at(i).vertices[vertex].x);
            boundingBoxMin.y = std::min(boundingBoxMin.y, meshes.at(i).vertices[vertex].y);
            boundingBoxMin.z = std::min(boundingBoxMin.z, meshes.at(i).vertices[vertex].z);
            boundingBoxMax.x = std::max(boundingBoxMax.x, meshes.at(i).vertices[vertex].x);
            boundingBoxMax.y = std::max(boundingBoxMax.y, meshes.at(i).vertices[vertex].y);
            boundingBoxMax.z = std::max(boundingBoxMax.z, meshes.at(i).vertices[vertex].z);
        }
    }

    float3 boundingBoxDimensions = make_float3(
            boundingBoxMax.x - boundingBoxMin.x,
            boundingBoxMax.y - boundingBoxMin.y,
            boundingBoxMax.z - boundingBoxMin.z);
    float largestBoundingBoxSide = std::max(std::max(boundingBoxDimensions.x, boundingBoxDimensions.y), boundingBoxDimensions.z);

    // Each recursion level splits up the lowest level nodes into 28 smaller ones.
    // This regularity means we can calculate the total number of objects we need to render
    // which we can of course preallocate
    unsigned long totalItemsToRender = 0;
    for(unsigned long level = 0; level < depthLimit; level++) {
        totalItemsToRender += std::pow(26ul, level);
    }

    workItemGPU* workQueue = new workItemGPU[totalItemsToRender];
    std::cout << "Number of items to be rendered: " << totalItemsToRender << std::endl;

    // TRANSFER MESHES OM THE GPU

    GPUMesh* CPUMeshes = new GPUMesh[meshes.size()];
    GPUMesh* GPUMeshes = nullptr;
    // Allocate the Meshes array on the GPU
    checkCudaErrors(cudaMalloc((void **)&GPUMeshes, sizeof(GPUMesh) * meshes.size()));

    for (unsigned int i = 0; i < meshes.size(); i++)
    {
      // Allocate Vertices information on GPU
      float4* verticesGPU;
      checkCudaErrors(cudaMalloc((void **)&verticesGPU, sizeof(float4) * meshes.at(i).vertexCount));
      // Allocate Normals information on GPU
      float3* normalsGPU;
      checkCudaErrors(cudaMalloc((void **)&normalsGPU, sizeof(float3) * meshes.at(i).vertexCount));

      // Transfer vertices to the GPU
      checkCudaErrors(cudaMemcpy(verticesGPU, (float4*)meshes.at(i).vertices, sizeof(float4) * meshes.at(i).vertexCount, cudaMemcpyHostToDevice));
      // Transfer normals to the GPU
      checkCudaErrors(cudaMemcpy(normalsGPU, meshes.at(i).normals, sizeof(float3) * meshes.at(i).vertexCount, cudaMemcpyHostToDevice));

      // Store the device pointer on the CPU array of meshes
      CPUMeshes[i].vertices = verticesGPU;
      CPUMeshes[i].normals = normalsGPU;
      // Copy all the other field into the mesh stored on the CPU
      CPUMeshes[i].vertexCount = meshes.at(i).vertexCount;
      CPUMeshes[i].objectDiffuseColour = meshes.at(i).objectDiffuseColour;
      CPUMeshes[i].hasNormals = meshes.at(i).hasNormals;
    }

    // Transfer array of Meshes from CPU to GPU
    checkCudaErrors(cudaMemcpy(GPUMeshes, CPUMeshes, sizeof(GPUMesh) * meshes.size(), cudaMemcpyHostToDevice));

    // MESH ARRAY TESTS
    GPUMesh* GPUMeshTest = new GPUMesh[meshes.size()];
    checkCudaErrors(cudaMemcpy(GPUMeshTest, GPUMeshes, sizeof(GPUMesh) * meshes.size(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < meshes.size(); i++)
    {
      //std::cout << "VERTEX TEST :  " << GPUMeshTest[i].vertices[0].y << '\n';
      //std::cout << "VERTEX ORIGINAL: " << meshes.at(i).vertices[0].x <<'\n';
    }
    std::cout << "VERTEX COUNT ON THE ORIGINAL MESH : " <<  <<'\n';
    std::cout << "VERTEX COUNT ON THE ORIGINAL MESH : " << meshes.at(0).vertices[0].x <<'\n';
    std::cout << "VERTEX COUNT ON THE TEST MESH :  " << GPUMeshTest[0].vertices[0].x << '\n';


    unsigned long counter = 0;
    fillWorkQueue(workQueue, largestBoundingBoxSide, depthLimit, &counter);

    // Allocate workQueue into the GPU
    workItemGPU* workQueueGPU;
    checkCudaErrors(cudaMalloc((void **)&workQueueGPU, sizeof(workItemGPU) * totalItemsToRender));
    // Transfer from workQueue to workQueueGPU
    checkCudaErrors(cudaMemcpy(workQueueGPU, workQueue, sizeof(workItemGPU) * totalItemsToRender, cudaMemcpyHostToDevice));

    // WORKQUEUE TEST
    workItemGPU* workQueueGPUTest = new workItemGPU[totalItemsToRender];
    checkCudaErrors(cudaMemcpy(workQueueGPUTest , workQueueGPU, sizeof(workItemGPU) * totalItemsToRender, cudaMemcpyDeviceToHost));
    int counterWorkQueue = 0;
    for (int i = 0; i < totalItemsToRender; i++)
    {
      if (workQueueGPUTest[i].scale == workQueue[i].scale &&
          workQueueGPUTest[i].distanceOffset.x == workQueue[i].distanceOffset.x &&
          workQueueGPUTest[i].distanceOffset.y == workQueue[i].distanceOffset.y &&
          workQueueGPUTest[i].distanceOffset.z == workQueue[i].distanceOffset.z)
          counterWorkQueue++;
    }

    if (counterWorkQueue == totalItemsToRender) std::cout << "\n" << "WORK QUEUE TEST :  PASSED" << "\n" <<'\n';

	renderMeshes(
			totalItemsToRender, workQueue,
			meshes.data(), meshes.size(),
			width, height, frameBuffer, depthBuffer);


    std::cout << "Finished!" << std::endl;

    // Copy the output picture into a vector so that the image dump code is happy :)
    std::vector<unsigned char> outputFramebuffer(frameBuffer, frameBuffer + (width * height * 4));

    return outputFramebuffer;
}
