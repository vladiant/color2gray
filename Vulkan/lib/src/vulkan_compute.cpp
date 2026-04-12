#include "vulkan_compute.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

VulkanCompute::VulkanCompute(const std::string& shaderDir)
    : mShaderDir(shaderDir) {
  try {
    createInstance();
    pickPhysicalDevice();
    createLogicalDevice();
    createCommandPool();
    createDescriptorPool();
    createNQPipeline();
    createQPipeline();
    mInitialized = true;
    std::cout << "Vulkan compute initialized successfully.\n";
  } catch (const std::exception& e) {
    std::cerr << "Vulkan init failed: " << e.what()
              << "\nFalling back to CPU.\n";
    cleanup();
  }
}

VulkanCompute::~VulkanCompute() { cleanup(); }

void VulkanCompute::createInstance() {
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "color2gray";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  if (vkCreateInstance(&createInfo, nullptr, &mInstance) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan instance");
  }
}

void VulkanCompute::pickPhysicalDevice() {
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr);
  if (deviceCount == 0) {
    throw std::runtime_error("No Vulkan-capable GPU found");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(mInstance, &deviceCount, devices.data());

  // Pick a device with a compute queue
  for (const auto& dev : devices) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount,
                                             queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
      if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        mPhysicalDevice = dev;
        mComputeQueueFamily = i;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        std::cout << "Using GPU: " << props.deviceName << '\n';
        return;
      }
    }
  }

  throw std::runtime_error("No GPU with compute capability found");
}

void VulkanCompute::createLogicalDevice() {
  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = mComputeQueueFamily;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.queueCreateInfoCount = 1;
  createInfo.pQueueCreateInfos = &queueCreateInfo;

  if (vkCreateDevice(mPhysicalDevice, &createInfo, nullptr, &mDevice) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create logical device");
  }

  vkGetDeviceQueue(mDevice, mComputeQueueFamily, 0, &mComputeQueue);
}

void VulkanCompute::createCommandPool() {
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = mComputeQueueFamily;

  if (vkCreateCommandPool(mDevice, &poolInfo, nullptr, &mCommandPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create command pool");
  }
}

void VulkanCompute::createDescriptorPool() {
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 6;  // max 3 buffers * 2 pipelines

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  poolInfo.maxSets = 2;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;

  if (vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mDescriptorPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor pool");
  }
}

VkShaderModule VulkanCompute::loadShaderModule(const std::string& filename) {
  std::string path = mShaderDir + "/" + filename;
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open shader file: " + path);
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = buffer.size();
  createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(mDevice, &createInfo, nullptr, &shaderModule) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module from: " + path);
  }

  return shaderModule;
}

void VulkanCompute::createNQPipeline() {
  // Descriptor set layout: 2 storage buffers (lab data + output)
  VkDescriptorSetLayoutBinding bindings[2]{};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 2;
  layoutInfo.pBindings = bindings;

  if (vkCreateDescriptorSetLayout(mDevice, &layoutInfo, nullptr,
                                  &mDescSetLayoutNQ) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create NQ descriptor set layout");
  }

  // Push constant range
  VkPushConstantRange pushRange{};
  pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushRange.offset = 0;
  pushRange.size = sizeof(PushConstantsNQ);

  VkPipelineLayoutCreateInfo pipeLayoutInfo{};
  pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeLayoutInfo.setLayoutCount = 1;
  pipeLayoutInfo.pSetLayouts = &mDescSetLayoutNQ;
  pipeLayoutInfo.pushConstantRangeCount = 1;
  pipeLayoutInfo.pPushConstantRanges = &pushRange;

  if (vkCreatePipelineLayout(mDevice, &pipeLayoutInfo, nullptr,
                             &mPipelineLayoutNQ) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create NQ pipeline layout");
  }

  VkShaderModule shader = loadShaderModule("calc_d_nq.spv");

  VkComputePipelineCreateInfo pipeInfo{};
  pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeInfo.stage.module = shader;
  pipeInfo.stage.pName = "main";
  pipeInfo.layout = mPipelineLayoutNQ;

  if (vkCreateComputePipelines(mDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
                               &mPipelineNQ) != VK_SUCCESS) {
    vkDestroyShaderModule(mDevice, shader, nullptr);
    throw std::runtime_error("Failed to create NQ compute pipeline");
  }

  vkDestroyShaderModule(mDevice, shader, nullptr);
}

void VulkanCompute::createQPipeline() {
  // Descriptor set layout: 3 storage buffers (lab data + q data + output)
  VkDescriptorSetLayoutBinding bindings[3]{};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[2].binding = 2;
  bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[2].descriptorCount = 1;
  bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 3;
  layoutInfo.pBindings = bindings;

  if (vkCreateDescriptorSetLayout(mDevice, &layoutInfo, nullptr,
                                  &mDescSetLayoutQ) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Q descriptor set layout");
  }

  VkPushConstantRange pushRange{};
  pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushRange.offset = 0;
  pushRange.size = sizeof(PushConstantsQ);

  VkPipelineLayoutCreateInfo pipeLayoutInfo{};
  pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeLayoutInfo.setLayoutCount = 1;
  pipeLayoutInfo.pSetLayouts = &mDescSetLayoutQ;
  pipeLayoutInfo.pushConstantRangeCount = 1;
  pipeLayoutInfo.pPushConstantRanges = &pushRange;

  if (vkCreatePipelineLayout(mDevice, &pipeLayoutInfo, nullptr,
                             &mPipelineLayoutQ) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Q pipeline layout");
  }

  VkShaderModule shader = loadShaderModule("calc_d_q.spv");

  VkComputePipelineCreateInfo pipeInfo{};
  pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeInfo.stage.module = shader;
  pipeInfo.stage.pName = "main";
  pipeInfo.layout = mPipelineLayoutQ;

  if (vkCreateComputePipelines(mDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
                               &mPipelineQ) != VK_SUCCESS) {
    vkDestroyShaderModule(mDevice, shader, nullptr);
    throw std::runtime_error("Failed to create Q compute pipeline");
  }

  vkDestroyShaderModule(mDevice, shader, nullptr);
}

VulkanCompute::Buffer VulkanCompute::createBuffer(
    VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties) {
  Buffer buf;

  VkBufferCreateInfo bufInfo{};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size = size;
  bufInfo.usage = usage;
  bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(mDevice, &bufInfo, nullptr, &buf.buffer) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create buffer");
  }

  VkMemoryRequirements memReqs;
  vkGetBufferMemoryRequirements(mDevice, buf.buffer, &memReqs);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties);

  if (vkAllocateMemory(mDevice, &allocInfo, nullptr, &buf.memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(mDevice, buf.buffer, nullptr);
    throw std::runtime_error("Failed to allocate buffer memory");
  }

  vkBindBufferMemory(mDevice, buf.buffer, buf.memory, 0);
  return buf;
}

void VulkanCompute::destroyBuffer(Buffer& buf) {
  if (buf.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(mDevice, buf.buffer, nullptr);
    buf.buffer = VK_NULL_HANDLE;
  }
  if (buf.memory != VK_NULL_HANDLE) {
    vkFreeMemory(mDevice, buf.memory, nullptr);
    buf.memory = VK_NULL_HANDLE;
  }
}

uint32_t VulkanCompute::findMemoryType(uint32_t typeFilter,
                                       VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memProps);

  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) &&
        (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find suitable memory type");
}

void VulkanCompute::submitAndWait(VkCommandBuffer cmdBuf) {
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuf;

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

  VkFence fence;
  if (vkCreateFence(mDevice, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create fence");
  }

  if (vkQueueSubmit(mComputeQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
    vkDestroyFence(mDevice, fence, nullptr);
    throw std::runtime_error("Failed to submit compute command buffer");
  }

  vkWaitForFences(mDevice, 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(mDevice, fence, nullptr);
}

void VulkanCompute::calcDNQ(int N, const float* labData, float theta,
                            float alpha, float* result) {
  const VkDeviceSize labSize = static_cast<VkDeviceSize>(N) * 4 * sizeof(float);
  const VkDeviceSize dSize = static_cast<VkDeviceSize>(N) * sizeof(float);

  const auto memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  Buffer labBuf = createBuffer(labSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               memFlags);
  Buffer dBuf = createBuffer(dSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             memFlags);

  // Upload lab data
  void* mapped = nullptr;
  vkMapMemory(mDevice, labBuf.memory, 0, labSize, 0, &mapped);
  std::memcpy(mapped, labData, labSize);
  vkUnmapMemory(mDevice, labBuf.memory);

  // Zero output buffer
  vkMapMemory(mDevice, dBuf.memory, 0, dSize, 0, &mapped);
  std::memset(mapped, 0, dSize);
  vkUnmapMemory(mDevice, dBuf.memory);

  // Allocate descriptor set
  VkDescriptorSet descriptorSet;
  VkDescriptorSetAllocateInfo dsAllocInfo{};
  dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAllocInfo.descriptorPool = mDescriptorPool;
  dsAllocInfo.descriptorSetCount = 1;
  dsAllocInfo.pSetLayouts = &mDescSetLayoutNQ;

  if (vkAllocateDescriptorSets(mDevice, &dsAllocInfo, &descriptorSet) !=
      VK_SUCCESS) {
    destroyBuffer(labBuf);
    destroyBuffer(dBuf);
    throw std::runtime_error("Failed to allocate NQ descriptor set");
  }

  // Update descriptor set
  VkDescriptorBufferInfo labBufInfo{};
  labBufInfo.buffer = labBuf.buffer;
  labBufInfo.offset = 0;
  labBufInfo.range = labSize;

  VkDescriptorBufferInfo dBufInfo{};
  dBufInfo.buffer = dBuf.buffer;
  dBufInfo.offset = 0;
  dBufInfo.range = dSize;

  VkWriteDescriptorSet writes[2]{};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = descriptorSet;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &labBufInfo;

  writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet = descriptorSet;
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &dBufInfo;

  vkUpdateDescriptorSets(mDevice, 2, writes, 0, nullptr);

  uint32_t groupCount = (static_cast<uint32_t>(N) + 255) / 256;

  // Tile inner loop to avoid GPU timeout.
  // Each tile processes a range [tileStart, tileEnd) of the inner j loop.
  constexpr uint32_t kTileSize = 4096;
  uint32_t totalPixels = static_cast<uint32_t>(N);

  for (uint32_t tileStart = 0; tileStart < totalPixels;
       tileStart += kTileSize) {
    uint32_t tileEnd = std::min(tileStart + kTileSize, totalPixels);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = mCommandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuf;
    vkAllocateCommandBuffers(mDevice, &cmdAllocInfo, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, mPipelineNQ);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            mPipelineLayoutNQ, 0, 1, &descriptorSet, 0,
                            nullptr);

    PushConstantsNQ pc{};
    pc.N = totalPixels;
    pc.theta = theta;
    pc.alpha = alpha;
    pc.tileStart = tileStart;
    pc.tileEnd = tileEnd;
    vkCmdPushConstants(cmdBuf, mPipelineLayoutNQ, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(cmdBuf, groupCount, 1, 1);

    vkEndCommandBuffer(cmdBuf);

    submitAndWait(cmdBuf);

    vkFreeCommandBuffers(mDevice, mCommandPool, 1, &cmdBuf);
  }

  // Read results
  vkMapMemory(mDevice, dBuf.memory, 0, dSize, 0, &mapped);
  std::memcpy(result, mapped, dSize);
  vkUnmapMemory(mDevice, dBuf.memory);

  // Cleanup per-call resources
  vkFreeDescriptorSets(mDevice, mDescriptorPool, 1, &descriptorSet);
  destroyBuffer(labBuf);
  destroyBuffer(dBuf);
}

void VulkanCompute::calcDQ(int N, const float* labData, int qSize,
                           const float* qData, float theta, float alpha,
                           float* result) {
  const VkDeviceSize labSize = static_cast<VkDeviceSize>(N) * 4 * sizeof(float);
  const VkDeviceSize qBufSize =
      static_cast<VkDeviceSize>(qSize) * 4 * sizeof(float);
  const VkDeviceSize dSize = static_cast<VkDeviceSize>(N) * sizeof(float);

  const auto memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  Buffer labBuf = createBuffer(labSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               memFlags);
  Buffer qBuf = createBuffer(qBufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             memFlags);
  Buffer dBuf = createBuffer(dSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             memFlags);

  // Upload data
  void* mapped = nullptr;
  vkMapMemory(mDevice, labBuf.memory, 0, labSize, 0, &mapped);
  std::memcpy(mapped, labData, labSize);
  vkUnmapMemory(mDevice, labBuf.memory);

  vkMapMemory(mDevice, qBuf.memory, 0, qBufSize, 0, &mapped);
  std::memcpy(mapped, qData, qBufSize);
  vkUnmapMemory(mDevice, qBuf.memory);

  // Zero output buffer
  vkMapMemory(mDevice, dBuf.memory, 0, dSize, 0, &mapped);
  std::memset(mapped, 0, dSize);
  vkUnmapMemory(mDevice, dBuf.memory);

  // Allocate descriptor set
  VkDescriptorSet descriptorSet;
  VkDescriptorSetAllocateInfo dsAllocInfo{};
  dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAllocInfo.descriptorPool = mDescriptorPool;
  dsAllocInfo.descriptorSetCount = 1;
  dsAllocInfo.pSetLayouts = &mDescSetLayoutQ;

  if (vkAllocateDescriptorSets(mDevice, &dsAllocInfo, &descriptorSet) !=
      VK_SUCCESS) {
    destroyBuffer(labBuf);
    destroyBuffer(qBuf);
    destroyBuffer(dBuf);
    throw std::runtime_error("Failed to allocate Q descriptor set");
  }

  // Update descriptor set
  VkDescriptorBufferInfo labBufInfo{};
  labBufInfo.buffer = labBuf.buffer;
  labBufInfo.offset = 0;
  labBufInfo.range = labSize;

  VkDescriptorBufferInfo qBufInfo{};
  qBufInfo.buffer = qBuf.buffer;
  qBufInfo.offset = 0;
  qBufInfo.range = qBufSize;

  VkDescriptorBufferInfo dBufInfo{};
  dBufInfo.buffer = dBuf.buffer;
  dBufInfo.offset = 0;
  dBufInfo.range = dSize;

  VkWriteDescriptorSet writes[3]{};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = descriptorSet;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &labBufInfo;

  writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet = descriptorSet;
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &qBufInfo;

  writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[2].dstSet = descriptorSet;
  writes[2].dstBinding = 2;
  writes[2].descriptorCount = 1;
  writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[2].pBufferInfo = &dBufInfo;

  vkUpdateDescriptorSets(mDevice, 3, writes, 0, nullptr);

  uint32_t groupCount = (static_cast<uint32_t>(N) + 255) / 256;

  // Tile inner loop to avoid GPU timeout
  constexpr uint32_t kTileSize = 4096;
  uint32_t totalQ = static_cast<uint32_t>(qSize);

  for (uint32_t tileStart = 0; tileStart < totalQ; tileStart += kTileSize) {
    uint32_t tileEnd = std::min(tileStart + kTileSize, totalQ);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = mCommandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuf;
    vkAllocateCommandBuffers(mDevice, &cmdAllocInfo, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, mPipelineQ);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            mPipelineLayoutQ, 0, 1, &descriptorSet, 0,
                            nullptr);

    PushConstantsQ pc{};
    pc.N = static_cast<uint32_t>(N);
    pc.qSize = totalQ;
    pc.theta = theta;
    pc.alpha = alpha;
    pc.tileStart = tileStart;
    pc.tileEnd = tileEnd;
    vkCmdPushConstants(cmdBuf, mPipelineLayoutQ, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(cmdBuf, groupCount, 1, 1);

    vkEndCommandBuffer(cmdBuf);

    submitAndWait(cmdBuf);

    vkFreeCommandBuffers(mDevice, mCommandPool, 1, &cmdBuf);
  }

  // Read results
  vkMapMemory(mDevice, dBuf.memory, 0, dSize, 0, &mapped);
  std::memcpy(result, mapped, dSize);
  vkUnmapMemory(mDevice, dBuf.memory);

  // Cleanup per-call resources
  vkFreeDescriptorSets(mDevice, mDescriptorPool, 1, &descriptorSet);
  destroyBuffer(labBuf);
  destroyBuffer(qBuf);
  destroyBuffer(dBuf);
}

void VulkanCompute::cleanup() {
  if (mDevice != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(mDevice);

    if (mPipelineNQ != VK_NULL_HANDLE)
      vkDestroyPipeline(mDevice, mPipelineNQ, nullptr);
    if (mPipelineLayoutNQ != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(mDevice, mPipelineLayoutNQ, nullptr);
    if (mDescSetLayoutNQ != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(mDevice, mDescSetLayoutNQ, nullptr);

    if (mPipelineQ != VK_NULL_HANDLE)
      vkDestroyPipeline(mDevice, mPipelineQ, nullptr);
    if (mPipelineLayoutQ != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(mDevice, mPipelineLayoutQ, nullptr);
    if (mDescSetLayoutQ != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(mDevice, mDescSetLayoutQ, nullptr);

    if (mDescriptorPool != VK_NULL_HANDLE)
      vkDestroyDescriptorPool(mDevice, mDescriptorPool, nullptr);
    if (mCommandPool != VK_NULL_HANDLE)
      vkDestroyCommandPool(mDevice, mCommandPool, nullptr);

    vkDestroyDevice(mDevice, nullptr);
  }

  if (mInstance != VK_NULL_HANDLE) {
    vkDestroyInstance(mInstance, nullptr);
  }

  mDevice = VK_NULL_HANDLE;
  mInstance = VK_NULL_HANDLE;
  mInitialized = false;
}
