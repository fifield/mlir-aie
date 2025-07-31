//===- AIETargetSimulationFiles.cpp -----------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Targets/AIETargets.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "llvm/IR/Module.h"

using namespace xilinx;
using namespace AIE;

mlir::LogicalResult AIE::AIETranslateSCSimConfig(mlir::ModuleOp module,
                                                 llvm::raw_ostream &output,
                                                 llvm::StringRef deviceName) {
  DeviceOp deviceOp =
      AIE::DeviceOp::getForSymbolInModuleOrError(module, deviceName);
  if (!deviceOp) {
    return mlir::failure();
  }
  AIEArch arch = deviceOp.getTargetModel().getTargetArch();

  if (arch == AIEArch::AIE4 || arch == AIEArch::AIE4z) {
    output << "{\n"
           << "    \"SimulationConfig\": {\n"
           << "        \"device_json\": {\n"
           << "            \"directory\": \"data/aie4/devices\",\n"
           << "            \"file\": \"aie4b_4r_3c_1m_medusa.json\"\n"
           << "        },\n"
           << "        \"phy_device_file\": \"xc10MDS1-die-0x-e-S-es0\",\n"
           << "        \"aiearch\": \"aie4\",\n"
           << "        \"aie_freq\": 1000000000.0,\n"
           << "        \"use_real_noc\": 1,\n"
           << "        \"evaluate_fifo_depth\": 0,\n"
           << "        \"shim_sol\": \"arch/aieshim_solution.aiesol\",\n"
           << "        \"pl_ip_block\": [\n"
           << "            {\n"
           << "                \"name\": \"ps_ps_main\",\n"
           << "                \"ip\": \"ps\",\n"
           << "                \"lib_path\": \"ps/ps.so\",\n"
           << "                \"pl_freq\": 250000000.0,\n"
           << "                \"axi_mm\": [\n"
           << "                    {\n"
           << "                        \"port_name\": \"ps_axi\",\n"
           << "                        \"direction\": \"ps_to_gm\",\n"
           << "                        \"bus_width\": 0\n"
           << "                    }\n"
           << "                ],\n"
           << "                \"event_bus\": []\n"
           << "            }\n"
           << "        ]\n"
           << "    }\n"
           << "}\n";
  } else if (arch == AIEArch::AIE2p) {
    output << "{\n"
           << "    \"SimulationConfig\": {\n"
           << "        \"device_json\": {\n"
           << "            \"directory\": \"data/aie2p/devices\",\n"
           << "            \"file\": \"aie2p_8x4_device.json\"\n"
           << "        },\n"
           << "        \"phy_device_file\": \"aie2p_8x4_device\",\n"
           << "        \"aiearch\": \"aie2p\",\n"
           << "        \"aie_freq\": 1000000000.0,\n"
           << "        \"use_real_noc\": 1,\n"
           << "        \"evaluate_fifo_depth\": 0,\n"
           << "        \"noc_ip_block\": {\n"
           << "            \"lib_path\": \"./sim/noc/liblnoc_tlm.so\",\n"
           << "            \"traffic_file\": \"./sim/noc/noc_traffic.nts\",\n"
           << "            \"config_file\": \"./sim/noc/noc_soln.ncr\"\n"
           << "        },\n"
           << "        \"pl_ip_block\": [\n"
           << "            {\n"
           << "                \"name\": \"ps_ps_main\",\n"
           << "                \"ip\": \"ps\",\n"
           << "                \"lib_path\": \"ps/ps.so\",\n"
           << "                \"pl_freq\": 362500000.0,\n"
           << "                \"axi_mm\": [\n"
           << "                    {\n"
           << "                        \"port_name\": \"ps_axi\",\n"
           << "                        \"direction\": \"ps_to_gm\",\n"
           << "                        \"noc_endpoint\": \"NOC_NMU128_X0Y5\",\n"
           << "                        \"bus_width\": 0\n"
           << "                    }\n"
           << "                ],\n"
           << "                \"event_bus\": []\n"
           << "            }\n"
           << "        ]\n"
           << "    }\n"
           << "}\n";
  } else if (arch == AIEArch::AIE2) {
    output << "{\n"
           << "    \"SimulationConfig\": {\n"
           << "        \"device_json\": {\n"
           << "            \"directory\": \"data/aie_ml/devices\",\n";
    if (deviceOp.getTargetModel().hasProperty(AIETargetModel::IsNPU)) {
      output << "            \"file\": \"aie2_5x4_device.json\"\n"
             << "        },\n"
             << "        \"phy_device_file\": \"aie2_5x4_device\",\n";
    } else {
      output << "            \"file\": \"VC2802.json\"\n"
             << "        },\n"
             << "        \"phy_device_file\": \"aie2_5x4_device\",\n";
    }
    output << "        \"aiearch\": \"aie2\",\n"
           << "        \"aie_freq\": 1050000000.0,\n"
           << "        \"use_real_noc\": 1,\n"
           << "        \"evaluate_fifo_depth\": 0,\n"
           << "        \"noc_ip_block\": {\n"
           << "            \"lib_path\": \"./sim/noc/liblnoc_tlm.so\",\n"
           << "            \"traffic_file\": \"./sim/noc/noc_traffic.nts\",\n"
           << "            \"config_file\": \"./sim/noc/noc_soln.ncr\"\n"
           << "        },\n"
           << "        \"pl_ip_block\": [\n"
           << "            {\n"
           << "                \"name\": \"ps_ps_main\",\n"
           << "                \"ip\": \"ps\",\n"
           << "                \"lib_path\": \"ps/ps.so\",\n"
           << "                \"pl_freq\": 362500000.0,\n"
           << "                \"axi_mm\": [\n"
           << "                    {\n"
           << "                        \"port_name\": \"ps_axi\",\n"
           << "                        \"direction\": \"ps_to_gm\",\n"
           << "                        \"noc_endpoint\": \"NOC_NMU128_X0Y5\",\n"
           << "                        \"bus_width\": 0\n"
           << "                    }\n"
           << "                ],\n"
           << "                \"event_bus\": []\n"
           << "            }\n"
           << "        ]\n"
           << "    }\n"
           << "}\n";
  } else if (arch == AIEArch::AIE1) { // AIEArch::AIE1
    output << "{\n"
           << "    \"SimulationConfig\": {\n"
           << "        \"device_json\": {\n"
           << "            \"directory\": \"data/devices\",\n"
           << "            \"file\": \"VC1902.json\"\n"
           << "        },\n"
           << "        \"phy_device_file\": \"xcvc1902-vsva2197-2MP-e-S\",\n"
           << "        \"aiearch\": \"aie\",\n"
           << "        \"aie_freq\": 1250000000.0,\n"
           << "        \"use_real_noc\": 1,\n"
           << "        \"evaluate_fifo_depth\": 0,\n"
           << "        \"noc_ip_block\": {\n"
           << "            \"lib_path\": \"./Work/noc/liblnoc_tlm.so\",\n"
           << "            \"traffic_file\": \"./Work/noc/noc_traffic.nts\",\n"
           << "            \"config_file\": \"./Work/noc/noc_soln.ncr\"\n"
           << "        },\n"
           << "        \"pl_ip_block\": [\n"
           << "            {\n"
           << "                \"name\": \"ps_ps_main\",\n"
           << "                \"ip\": \"ps\",\n"
           << "                \"lib_path\": \"ps/ps.so\",\n"
           << "                \"pl_freq\": 312500000.0,\n"
           << "                \"axi_mm\": [\n"
           << "                    {\n"
           << "                        \"port_name\": \"ps_axi\",\n"
           << "                        \"direction\": \"ps_to_gm\",\n"
           << "                        \"noc_endpoint\": \"NOC_NMU128_X0Y5\",\n"
           << "                        \"bus_width\": 0\n"
           << "                    }\n"
           << "                ],\n"
           << "                \"event_bus\": []\n"
           << "            }\n"
           << "        ]\n"
           << "    }\n"
           << "}\n";
  }
  return mlir::success();
}

/* Generates a aieshim_solution.aiesol file which is necessary to run aiesim.
Sample invocation:
aie-translate --aie-mlir-to-shim-solution ./aie.mlir >
./Work/arch/aieshim_solution.aiesol

NOTE: to correctly enable all tiles used for routing, the aie-opt routing pass
must be called first. So, a more practical invocation: aie-opt
--aie-create-pathfinder-flows ./aie.mlir | aie-translate --aie-mlir-to-shim >
./Work/arch/aieshim_solution.aiesol
*/
mlir::LogicalResult AIE::AIETranslateShimSolution(mlir::ModuleOp module,
                                                  llvm::raw_ostream &output,
                                                  llvm::StringRef deviceName) {
  DeviceOp deviceOp =
      AIE::DeviceOp::getForSymbolInModuleOrError(module, deviceName);
  if (!deviceOp) {
    return mlir::failure();
  }
  if (failed(verifyDMAChannelsResolved(targetOp)))
    return mlir::failure();

  std::set<std::pair<uint32_t, uint32_t>> visitedPorts;

  // Generate boilerplate header
  output << "{\n";
  output << "  \"Placement\": [\n";

  // For each DMAStartOp in shims, generate a "LogicalInstance" section
  for (ShimDMAOp shimOp : deviceOp.getOps<ShimDMAOp>()) {
    for (DMAStartOp startOp : shimOp.getOps<DMAStartOp>()) {
      (void)startOp;
      int32_t column = shimOp.colIndex();
      // limit instances to one per column by setting channel=0 here, otherwise
      // there are crashes with aiesimulator for aie4
      int32_t channel = 0;
      if (visitedPorts.count({column, channel}))
        continue;

      int32_t count = visitedPorts.size();
      if (count > 0)
        output << ",\n";

      visitedPorts.insert({column, channel});

      // For aiesimulator to run, PortName must start at 00 and increase
      std::string port_name = "M";
      port_name += count < 10 ? "0" : "";
      port_name += std::to_string(count);
      port_name += "_AXI";

      // Generate a Logical Instance line
      output << "    {\n"
             << "      \"LogicalInstance\" : { \"InstanceName\" : "
             << "\"aie_engine_0\", \"PortName\" : \"" << port_name << "\"},\n";

      // "name" field appears to be arbitrary, but we try to be descriptive
      std::string physical_name = "";
      physical_name += "AIE_NOC_X" + std::to_string(column) + "Y0_AIE_NOC_";
      physical_name += "S_AXI_ch" + std::to_string(channel);

      // Generate a Physical Instance line
      output << "      \"PhysicalInstance\" : [{ \"name\" : \"" << physical_name
             << "\", \"column\" : " << column << ", \"channel\" : " << channel
             << " }],\n"
             << "      \"IsSoft\" : true\n    }";
    }
  }

  output << "\n  ]\n";
  output << "}\n";

  return mlir::success();
}

mlir::LogicalResult AIE::AIETranslateGraphXPE(mlir::ModuleOp module,
                                              llvm::raw_ostream &output,
                                              llvm::StringRef deviceName) {
  /* Generates a .xpe file which is necessary to run aiesim.
  .xpe is a power report file, but has information on which AIE tiles are used.
  Sample invocation:
  aie-translate --aie-mlir-to-xpe ./aie.mlir > ./Work/reports/graph.xpe

  NOTE: to correctly enable all tiles used for routing, the aie-opt routing pass
  must be called first. So, a more practical invocation: aie-opt
  --aie-create-pathfinder-flows ./aie.mlir | aie-translate --aie-mlir-to-xpe >
  ./Work/reports/graph.xpe
  */

  DeviceOp deviceOp =
      AIE::DeviceOp::getForSymbolInModuleOrError(module, deviceName);
  if (!deviceOp) {
    return mlir::failure();
  }
  const AIETargetModel &target = deviceOp.getTargetModel();

  // Generate boilerplate header
  // TODO: date and version should probably not be hardcoded
  output << "<?xml version=\"1.0\"?>"
         << "\n";
  output << "<POWERDATA data=\"AI-Engine Compiler\" dataVersion=\"2022.2\" "
            "design=\"graph\" date=\"2023\">\n";
  if (llvm::isa<AIE4TargetModel>(target)) {
    // AIE4 xc10MDS1
    output << " <DEVICE part=\"xc10MDS1\" grade=\"extended\" package=\"die\" "
              "speed=\"-0x\" process=\"typical\" vid=\"No\"></DEVICE>\n";
  } else if (llvm::isa<AIE2TargetModel>(target)) {
    output
        // AIE2 xcve2802
        << " <DEVICE part=\"xcve2802\" grade=\"extended\" package=\"vsvh1760\" "
           "speed=\"-2LP\" process=\"typical\" vid=\"No\"></DEVICE>\n";
    // AIE2 - xcve2302
    // << " <DEVICE part=\"xcve2302\" grade=\"extended\" package=\"sfva784\" "
    //    "speed=\"-1MP\" process=\"typical\" vid=\"No\"></DEVICE>\n";
  } else if (llvm::isa<AIE1TargetModel>(target)) { // AIEArch::AIE1
    output
        << " <DEVICE part=\"xcvc1902\" grade=\"extended\" package=\"vsva2197\" "
           "speed=\"-2MP\" process=\"typical\" vid=\"No\"></DEVICE>\n";
  } else {
    return deviceOp.emitOpError("unsupported aie.device target");
  }
  output << "  <AIE status=\"COMPILER_OUTPUT\">\n";

  // Generate design specific info on tiles within the mlir module
  auto module_tile_ops = deviceOp.getOps<TileOp>();
  int num_tiles = std::distance(module_tile_ops.begin(), module_tile_ops.end());
  // TODO: clk_freq only 1150 for AIE2
  if (llvm::isa<AIE4TargetModel>(target)) {
    output << R"(    <AIE_MODULE name="graph" num_tiles=")"
           << std::to_string(num_tiles) << "\" clk_freq=\"1000\">\n";
  } else if (llvm::isa<AIE2TargetModel>(target)) {
    output << R"(    <AIE_MODULE name="graph" num_tiles=")"
           << std::to_string(num_tiles) << "\" clk_freq=\"1150\">\n";
  } else if (llvm::isa<AIE1TargetModel>(target)) {
    output << R"(    <AIE_MODULE name="graph" num_tiles=")"
           << std::to_string(num_tiles) << "\" clk_freq=\"1250\">\n";
  } else {
    return deviceOp.emitOpError("unsupported aie.device target");
  }

  // Get all CoreOps into a convenient map which can then be referenced by
  // coordinates
  std::map<std::pair<int, int>, std::vector<CoreOp>> coreMap;
  for (CoreOp coreOp : deviceOp.getOps<CoreOp>())
    coreMap[std::make_pair(coreOp.colIndex(), coreOp.rowIndex())].push_back(
        coreOp);

  // For each TileOp in the module, generate a <TILE> section
  int kernel_count = 0;
  for (TileOp tileOp : module_tile_ops) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();

    // NOTE: row == 0 assumes shim always row 0
    if (tileOp.isShimNOCorPLTile() || tileOp.isMemTile())
      continue; // Skip shim and mem tiles (handled below)

    if (llvm::isa<AIE4TargetModel>(target)) {

      output << "      <TILE name=\"CR(" <<
          // CR coordinates ignores shim, and 2 mem rows hence row-3
          // AIE2 - xcve2802
          std::to_string(col) << ","
             << std::to_string(
                    row - deviceOp.getTargetModel().getNumMemTileRows() - 1)
             << ")\" "
             // CR coordinates ignores shim and 1 mem row, hence row-2
             // AIE2 - xcve2302
             // std::to_string(col) << "," << std::to_string(row - 2) << ")\" "
             << "type=\"int16\" int_core_load=\"1.0\" fp_core_load=\"0\" "
             << "mem_banks=\"0\" mem_rw_rate=\"0.2\" stream_util=\"0.0\" "
                "coordinates=\""
             <<
          // TODO: where does number of mem_banks come from?? Hardcoded to 0 for
          // now
          // TODO: does stream_util matter for sim?
          std::to_string(col) << ","
             << std::to_string(row -
                               deviceOp.getTargetModel().getNumMemTileRows())
             << "\">\n";

    } else if (llvm::isa<AIE2TargetModel>(target)) {

      output << "      <TILE name=\"CR(" <<
          // CR coordinates ignores shim, and 2 mem rows hence row-3
          // AIE2 - xcve2802
          std::to_string(col) << ","
             << std::to_string(
                    row - deviceOp.getTargetModel().getNumMemTileRows() - 1)
             << ")\" "
             // CR coordinates ignores shim and 1 mem row, hence row-2
             // AIE2 - xcve2302
             // std::to_string(col) << "," << std::to_string(row - 2) << ")\" "
             << R"(type="int16" int_core_load="1.0" fp_core_load="0" )"
             << "mem_banks=\"0\" mem_rw_rate=\"0.2\" stream_util=\"0.0\" "
                "coordinates=\""
             <<
          // TODO: where does number of mem_banks come from?? Hardcoded to 0 for
          // now
          // TODO: does stream_util matter for sim?
          std::to_string(col) << ","
             << std::to_string(row -
                               deviceOp.getTargetModel().getNumMemTileRows())
             << "\">\n";

    } else if (llvm::isa<AIE1TargetModel>(target)) {
      output << "      <TILE name=\"CR(" <<
          // CR coordinates ignores shim, hence row-1
          std::to_string(col) << "," << std::to_string(row - 1) << ")\" "
             << R"(type="int16" int_core_load="1.0" fp_core_load="0" )"
             << "mem_banks=\"0\" mem_rw_rate=\"0.2\" stream_util=\"0.0\" "
                "coordinates=\""
             <<
          // TODO: where does number of mem_banks come from?? Hardcoded to 0 for
          // now
          // TODO: does stream_util matter for sim?
          std::to_string(col) << "," << std::to_string(row) << "\">\n";
    } else {
      return deviceOp.emitOpError("unsupported aie.device target");
    }

    // If the TileOp has associated Kernels, generate <KERNEL> sections
    auto coreOps_in_tile =
        coreMap[std::make_pair(tileOp.colIndex(), tileOp.rowIndex())];
    for (auto coreOp : coreOps_in_tile) {
      (void)(coreOp); // get around unused variable warning for coreOp
      output << "        <KERNEL name=\"i" << std::to_string(kernel_count++)
             << "\" "
             << "int_core_load=\"" << std::to_string(1 / coreOps_in_tile.size())
             << "\" fp_core_load=\"0\"></KERNEL>\n";
    }
    output << "      </TILE>\n";
  }

  // For each ShimOp in the module, generate a <SHIM> section
  for (ShimDMAOp shimOp : deviceOp.getOps<ShimDMAOp>()) {
    if (llvm::isa<AIE4TargetModel>(target)) {
      const auto *noc_label = (deviceOp.getTargetModel().isShimNOCTile(
                                  shimOp.colIndex(), shimOp.rowIndex()))
                                  ? "AIE_PL_NOC_SIM"
                                  : "AIE_PL_SHIM";
      output << "      <SHIM name=\"SHIM(" << std::to_string(shimOp.colIndex())
             << ", " << std::to_string(shimOp.rowIndex())
             << ")\" "
                // TODO: stream_util can be 0 for aiesim purposes?
                "type=\""
             << noc_label << "\" stream_util=\"0\" num_pl_streams=\"0\" " <<
          // TODO: how to get num_aximm_connections from mlir?
          "num_aximm_connections=\"1\" coordinates=\""
             << std::to_string(shimOp.colIndex()) << ","
             << std::to_string(shimOp.rowIndex()) << "\" "
             << "></SHIM>\n";
    } else if (llvm::isa<AIE2TargetModel>(target)) {
      const auto *noc_label = (deviceOp.getTargetModel().isShimNOCTile(
                                  shimOp.colIndex(), shimOp.rowIndex()))
                                  ? "AIE_PL_NOC_SIM"
                                  : "AIE_PL_SHIM";
      output << "      <SHIM name=\"SHIM(" << std::to_string(shimOp.colIndex())
             << ", " << std::to_string(shimOp.rowIndex())
             << ")\" "
                // TODO: stream_util can be 0 for aiesim purposes?
                "type=\""
             << noc_label << R"(" stream_util="0" num_pl_streams="0" )" <<
          // TODO: how to get num_aximm_connections from mlir?
          R"(num_aximm_connections="1" coordinates=")"
             << std::to_string(shimOp.colIndex()) << ","
             << std::to_string(shimOp.rowIndex()) << "\" "
             << "></SHIM>\n";
    } else if (llvm::isa<AIE1TargetModel>(target)) {
      output << "      <SHIM name=\"SHIM(" << std::to_string(shimOp.colIndex())
             << ", " << std::to_string(shimOp.rowIndex()) << ")\" " <<
          // TODO: stream_util can be 0 for aiesim purposes?
          R"(type="AIE_PL_NOC_SHIM" stream_util="0" num_pl_streams="0"
        )" <<
          // TODO: how to get num_aximm_connections from mlir?
          R"(num_aximm_connections="1" coordinates=")"
             << std::to_string(shimOp.colIndex()) << ","
             << std::to_string(shimOp.rowIndex()) << "\" "
             << "></SHIM>\n";
    } else {
      return deviceOp.emitOpError("unsupported aie.device target");
    }
  }

  // For each memTile
  if (llvm::isa<AIE4TargetModel>(target)) {
    for (TileOp tileOp : module_tile_ops) {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();

      // NOTE: row == 0 assumes shim always row 0
      if (row == 0 ||
          row > static_cast<int>(deviceOp.getTargetModel().getNumMemTileRows()))
        continue; // Skip regular tiles (handled above)

      output << "      <MEM name=\"MEM(" << std::to_string(col) << ", "
             << std::to_string(row - 1)
             << ")\" "
                // TODO: stream_util can be 0 for aiesim purposes?
                "type=\"AIE_MEM\" mem_banks=\"0\" mme_rw_rate=\"0.1\" "
             << "stream_util=\"0.1\" coordinates=\"" << std::to_string(col)
             << "," << std::to_string(row - 1) << "\" "
             << "></MEM>\n";
    }
  } else if (llvm::isa<AIE2TargetModel>(target)) {
    for (TileOp tileOp : module_tile_ops) {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();

      // NOTE: row == 0 assumes shim always row 0
      if (row == 0 ||
          row > static_cast<int>(deviceOp.getTargetModel().getNumMemTileRows()))
        continue; // Skip regular tiles (handled above)

      output << "      <MEM name=\"MEM(" << std::to_string(col) << ", "
             << std::to_string(row - 1)
             << ")\" "
                // TODO: stream_util can be 0 for aiesim purposes?
                "type=\"AIE_MEM\" mem_banks=\"0\" mme_rw_rate=\"0.1\" "
             << R"(stream_util="0.1" coordinates=")" << std::to_string(col)
             << "," << std::to_string(row - 1) << "\" "
             << "></MEM>\n";
    }
  }

  output << "    </AIE_MODULE>\n";
  output << "  </AIE>\n";
  output << "</POWERDATA>\n";

  return mlir::success();
}
