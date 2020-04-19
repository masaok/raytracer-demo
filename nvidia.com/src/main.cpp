/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// ImGui - standalone example application for Glfw + Vulkan, using programmable
// pipeline If you are new to ImGui, see examples/README.txt and documentation
// at the top of imgui.cpp.

#include <array>
#include <vulkan/vulkan.hpp>


#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "appbase_vkpp.hpp"
#include "commands_vkpp.hpp"
#include "context_vkpp.hpp"
#include "glm/gtx/transform.hpp"
#include "hello_vulkan.h"
#include "manipulator.h"
#include "utilities_vkpp.hpp"

static bool             g_ResizeWanted = false;
static int              g_winWidth = 1280, g_winHeight = 720;
static VkDescriptorPool g_imguiDescPool;

//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

static void checkVkResult(VkResult err)
{
  if(err == 0)
    return;
  printf("VkResult %d\n", err);
  if(err < 0)
    abort();
}

//////////////////////////////////////////////////////////////////////////
// GLFW Callback functions
//////////////////////////////////////////////////////////////////////////

static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static void onResizeCallback(GLFWwindow* window, int w, int h)
{
  UNUSED(window);
  CameraManip.setWindowSize(w, h);
  g_ResizeWanted = true;
  g_winWidth     = w;
  g_winHeight    = h;
}

static void onKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if(ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse)
  {
    return;
  }

  if(action == GLFW_RELEASE)  // Don't deal with key up
  {
    return;
  }

  if(key == GLFW_KEY_ESCAPE || key == 'Q')
  {
    glfwSetWindowShouldClose(window, 1);
  }
}

static void onMouseMoveCallback(GLFWwindow* window, double mouseX, double mouseY)
{
  if(ImGui::GetCurrentContext() != nullptr)
  {
    ImGuiIO& io = ImGui::GetIO();
    if(io.WantCaptureKeyboard || io.WantCaptureMouse)
    {
      return;
    }
  }

  using nv_helpers_vk::Manipulator;
  Manipulator::Inputs inputs;
  inputs.lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  inputs.mmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
  inputs.rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
  if(!inputs.lmb && !inputs.rmb && !inputs.mmb)
  {
    return;  // no mouse button pressed
  }

  inputs.ctrl  = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
  inputs.shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
  inputs.alt   = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS;

  CameraManip.mouseMove(static_cast<int>(mouseX), static_cast<int>(mouseY), inputs);
}

static void onMouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
  if(ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse)
  {
    return;
  }

  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);
  CameraManip.setMousePosition(static_cast<int>(xpos), static_cast<int>(ypos));
}

static void onScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
  if(ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse)
  {
    return;
  }
  CameraManip.wheel(static_cast<int>(yoffset));
}

//////////////////////////////////////////////////////////////////////////
// IMGUI Setup
//////////////////////////////////////////////////////////////////////////
static void setupImGui(nvvkpp::AppBase& appBase, GLFWwindow* window)
{
  // Create the pool information descriptor used by ImGui
  {
    using vkDT = vk::DescriptorType;
    std::vector<vk::DescriptorPoolSize> counters{{vkDT::eCombinedImageSampler, 2}};
    vk::DescriptorPoolCreateInfo        poolInfo = {};
    poolInfo.setPoolSizeCount(static_cast<uint32_t>(counters.size()));
    poolInfo.setPPoolSizes(counters.data());
    poolInfo.setMaxSets(static_cast<uint32_t>(counters.size()));

    vk::Device device(appBase.getDevice());
    g_imguiDescPool = device.createDescriptorPool(poolInfo);
  }


  // Setup Dear ImGui binding
  {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo imGuiInitInfo = {};
    imGuiInitInfo.Allocator                 = nullptr;
    imGuiInitInfo.DescriptorPool            = g_imguiDescPool;
    imGuiInitInfo.Device                    = appBase.getDevice();
    imGuiInitInfo.ImageCount                = (uint32_t)appBase.getFramebuffers().size();
    imGuiInitInfo.Instance                  = appBase.getInstance();
    imGuiInitInfo.MinImageCount             = (uint32_t)appBase.getFramebuffers().size();
    imGuiInitInfo.PhysicalDevice            = appBase.getPhysicalDevice();
    imGuiInitInfo.PipelineCache             = appBase.getPipelineCache();
    imGuiInitInfo.Queue                     = appBase.getQueue();
    imGuiInitInfo.QueueFamily               = appBase.getQueueFamily();
    imGuiInitInfo.CheckVkResultFn           = checkVkResult;

    ImGui_ImplVulkan_Init(&imGuiInitInfo, appBase.getRenderPass());

    // Setup style
    ImGui::StyleColorsDark();
  }

  // Upload Fonts
  {
    nvvkpp::SingleCommandBuffer cmdBufGen(appBase.getDevice(), appBase.getQueueFamily());

    auto cmdBuf = cmdBufGen.createCommandBuffer();
    ImGui_ImplVulkan_CreateFontsTexture(cmdBuf);
    cmdBufGen.flushCommandBuffer(cmdBuf);

    ImGui_ImplVulkan_DestroyFontUploadObjects();
  }
}

static void destroyImgui(const vk::Device& device)
{
  if(ImGui::GetCurrentContext() != nullptr)
  {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    device.destroyDescriptorPool(g_imguiDescPool);
  }
}

void renderUI(HelloVulkan& helloVk)
{
  static int item = 1;
  if(ImGui::Combo("Up Vector", &item, "X\0Y\0Z\0\0"))
  {
    glm::vec3 pos, eye, up;
    CameraManip.getLookat(pos, eye, up);
    up = glm::vec3(item == 0, item == 1, item == 2);
    CameraManip.setLookat(pos, eye, up);
  }
  ImGui::SliderFloat3("Light Position", &helloVk.m_pushConstant.lightPosition.x, -20.f, 20.f);
  ImGui::SliderFloat("Light Intensity", &helloVk.m_pushConstant.lightIntensity, 0.f, 100.f);
  ImGui::RadioButton("Point", &helloVk.m_pushConstant.lightType, 0);
  ImGui::SameLine();
  ImGui::RadioButton("Infinite", &helloVk.m_pushConstant.lightType, 1);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
  UNUSED(argc);
  UNUSED(argv);

  // Setup window
  glfwSetErrorCallback(onErrorCallback);
  if(!glfwInit())
  {
    return 1;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(g_winWidth, g_winHeight,
                                        "NVIDIA Vulkan Raytracing Tutorial", nullptr, nullptr);

  glfwSetScrollCallback(window, onScrollCallback);
  glfwSetKeyCallback(window, onKeyCallback);
  glfwSetCursorPosCallback(window, onMouseMoveCallback);
  glfwSetMouseButtonCallback(window, onMouseButtonCallback);
  glfwSetFramebufferSizeCallback(window, onResizeCallback);

  // Setup camera
  CameraManip.setWindowSize(g_winWidth, g_winHeight);
  CameraManip.setLookat(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

  // Setup Vulkan
  if(!glfwVulkanSupported())
  {
    printf("GLFW: Vulkan Not Supported\n");
    return 1;
  }

  // Enabling the extension feature
  vk::PhysicalDeviceDescriptorIndexingFeaturesEXT indexFeature;
  vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT  scalarFeature;

  // Requesting Vulkan extensions and layers
  nvvkpp::ContextCreateInfo contextInfo;
  contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);
  contextInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME);
  contextInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
  contextInfo.addInstanceExtension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, false, &indexFeature);
  contextInfo.addDeviceExtension(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME, false, &scalarFeature);

  // Creating Vulkan base application
  nvvkpp::AppBase appBase;
  appBase.setupVulkan(contextInfo, window);
  appBase.createSurface(g_winWidth, g_winHeight);
  appBase.createDepthBuffer();
  appBase.createRenderPass();
  appBase.createFrameBuffers();


  // Setup Imgui
  setupImGui(appBase, window);

  // Creation of the example
  HelloVulkan helloVk;
  helloVk.init(appBase.getDevice(), appBase.getPhysicalDevice(), appBase.getQueueFamily(),
               appBase.getSize());
  helloVk.loadModel("../media/scenes/cube_multi.obj");

  helloVk.createOffscreenRender();
  helloVk.createDescriptorSetLayout();
  helloVk.createGraphicsPipeline(appBase.getRenderPass());
  helloVk.createUniformBuffer();
  helloVk.createSceneDescriptionBuffer();
  helloVk.updateDescriptorSet();

  helloVk.createPostDescriptor();
  helloVk.createPostPipeline(appBase.getRenderPass());
  helloVk.updatePostDescriptorSet();
  glm::vec4 clearColor = glm::vec4(1, 1, 1, 1.00f);

  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();

    if(g_ResizeWanted)
    {
      appBase.onWindowResize(g_winWidth, g_winHeight);
      helloVk.resize(appBase.getSize());
    }
    g_ResizeWanted = false;

    helloVk.updateUniformBuffer();

    // Start the Dear ImGui frame
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Show a simple window.
    if(1 == 1)
    {
      // Edit 3 floats representing a color
      ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(&clearColor));

      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

      renderUI(helloVk);

      ImGui::Render();
    }


    // render the scene
    appBase.prepareFrame();

    auto                     curFrame = appBase.getCurFrame();
    const vk::CommandBuffer& cmdBuff  = appBase.getCommandBuffers()[curFrame];

    cmdBuff.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::ClearValue clearValues[2];
    clearValues[0].setColor(nvvkpp::util::clearColor(clearColor));
    clearValues[1].setDepthStencil({1.0f, 0});

    // Begin render pass
    vk::RenderPassBeginInfo offscreenRenderPassBeginInfo;
    offscreenRenderPassBeginInfo.setClearValueCount(2);
    offscreenRenderPassBeginInfo.setPClearValues(clearValues);
    offscreenRenderPassBeginInfo.setRenderPass(helloVk.m_offscreenRenderPass);
    offscreenRenderPassBeginInfo.setFramebuffer(helloVk.m_offscreenFramebuffer);
    offscreenRenderPassBeginInfo.setRenderArea({{}, appBase.getSize()});

    // Rendering Scene
    {
      cmdBuff.beginRenderPass(offscreenRenderPassBeginInfo, vk::SubpassContents::eInline);
      helloVk.rasterize(cmdBuff);
      cmdBuff.endRenderPass();
    }


    vk::RenderPassBeginInfo postRenderPassBeginInfo;
    postRenderPassBeginInfo.setClearValueCount(2);
    postRenderPassBeginInfo.setPClearValues(clearValues);
    postRenderPassBeginInfo.setRenderPass(appBase.getRenderPass());
    postRenderPassBeginInfo.setFramebuffer(appBase.getFramebuffers()[curFrame]);
    postRenderPassBeginInfo.setRenderArea({{}, appBase.getSize()});
    postRenderPassBeginInfo.setRenderPass(appBase.getRenderPass());
    postRenderPassBeginInfo.setFramebuffer(appBase.getFramebuffers()[curFrame]);
    cmdBuff.beginRenderPass(postRenderPassBeginInfo, vk::SubpassContents::eInline);
    helloVk.drawPost(cmdBuff);
    // Rendering UI
    {
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuff);
    }

    cmdBuff.endRenderPass();
    cmdBuff.end();
    appBase.submitFrame();
  }

  // Cleanup
  appBase.getDevice().waitIdle();
  destroyImgui(appBase.getDevice());

  helloVk.destroyResources();
  appBase.destroy();


  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
