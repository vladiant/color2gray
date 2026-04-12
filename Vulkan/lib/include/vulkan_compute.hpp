#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

class VulkanCompute {
 public:
  explicit VulkanCompute(const std::string& shaderDir);
  ~VulkanCompute();

  VulkanCompute(const VulkanCompute&) = delete;
  VulkanCompute& operator=(const VulkanCompute&) = delete;

  bool isAvailable() const { return mInitialized; }

  // Compute d[i] = sum_{j!=i} calc_delta(i,j) for non-quantized images.
  // labData: N elements of [l, a, b, 0] packed as 4 floats each.
  // result: N floats output.
  void calcDNQ(int N, const float* labData, float theta, float alpha,
               float* result);

  // Compute d[i] for quantized images.
  // labData: N elements of [l, a, b, 0] packed as 4 floats each.
  // qData: qSize elements of [l, a, b, count] packed as 4 floats each.
  // result: N floats output.
  void calcDQ(int N, const float* labData, int qSize, const float* qData,
              float theta, float alpha, float* result);

 private:
  struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
  };

  struct PushConstantsNQ {
    uint32_t N;
    float theta;
    float alpha;
    uint32_t tileStart;
    uint32_t tileEnd;
  };

  struct PushConstantsQ {
    uint32_t N;
    uint32_t qSize;
    float theta;
    float alpha;
    uint32_t tileStart;
    uint32_t tileEnd;
  };

  bool mInitialized = false;
  std::string mShaderDir;

  VkInstance mInstance = VK_NULL_HANDLE;
  VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
  VkDevice mDevice = VK_NULL_HANDLE;
  VkQueue mComputeQueue = VK_NULL_HANDLE;
  uint32_t mComputeQueueFamily = 0;
  VkCommandPool mCommandPool = VK_NULL_HANDLE;
  VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;

  // Non-quantized pipeline
  VkDescriptorSetLayout mDescSetLayoutNQ = VK_NULL_HANDLE;
  VkPipelineLayout mPipelineLayoutNQ = VK_NULL_HANDLE;
  VkPipeline mPipelineNQ = VK_NULL_HANDLE;

  // Quantized pipeline
  VkDescriptorSetLayout mDescSetLayoutQ = VK_NULL_HANDLE;
  VkPipelineLayout mPipelineLayoutQ = VK_NULL_HANDLE;
  VkPipeline mPipelineQ = VK_NULL_HANDLE;

  void createInstance();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createCommandPool();
  void createDescriptorPool();
  void createNQPipeline();
  void createQPipeline();
  void cleanup();

  VkShaderModule loadShaderModule(const std::string& filename);
  Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties);
  void destroyBuffer(Buffer& buf);
  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties);
  void submitAndWait(VkCommandBuffer cmdBuf);
};
