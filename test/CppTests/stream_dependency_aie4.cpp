// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/Transforms/AIEStreamDependencyAnalysis.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

static bool require(bool condition, const char *message) {
  if (!condition)
    llvm::errs() << message << "\n";
  return condition;
}

static bool checkVolumes(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      aie.device(npu3) {
        %shim = aie.tile(0, 0)
        %other = aie.tile(1, 0)
        %source_sw = aie.switchbox(%shim) {
          aie.connect<DMAControl : 0, East : 0>
        }
        %dest_sw = aie.switchbox(%other) {
          %a = aie.amsel<0> (0)
          aie.masterset(TileControl : 0, %a)
          aie.packet_rules(West : 0) {
            aie.rule(31, 7, %a)
            aie.rule(31, 9, %a)
            aie.rule(31, 1, %a)
          }
        }
        %buffer = aie.external_buffer : memref<64xi32>
        %dma = aie.shim_dma(%shim) {
          %data = aie.dma_start(MM2S, 0, ^data_bd, ^ctrl)
        ^data_bd:
          aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 1, pkt_type = 0>}
          aie.next_bd ^end
        ^ctrl:
          %control = aie.dma_start(ControlMM2S, 0, ^ctrl_bd, ^receive)
        ^ctrl_bd:
          aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 32) {packet = #aie.packet_info<pkt_id = 7, pkt_type = 0>}
          aie.next_bd ^ctrl_bd2
        ^ctrl_bd2:
          aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_id = 9, pkt_type = 0>}
          aie.next_bd ^end
        ^receive:
          %recv = aie.dma_start(S2MM, 0, ^recv_bd, ^trace)
        ^recv_bd:
          aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 8)
          aie.next_bd ^end
        ^trace:
          %tr = aie.dma_start(TraceS2MM, 0, ^trace_bd, ^end)
        ^trace_bd:
          aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 16)
          aie.next_bd ^end
        ^end:
          aie.end
        }
      }
    }
  )mlir",
                                            &context);
  if (!module)
    return false;
  auto device = *module->getOps<DeviceOp>().begin();
  RoutedStream data{
      {{0, 0}, {WireBundle::DMA, 0}}, {{0, 0}, {WireBundle::DMATrace, 0}}, 1};
  RoutedStream control{{{0, 0}, {WireBundle::DMAControl, 0}},
                       {{0, 2}, {WireBundle::TileControl, 0}},
                       7};
  RoutedStream controlThen = control;
  controlThen.packetID = 9;
  SmallVector<RoutedStream, 4> streams{data, control, controlThen};
  StreamVolumeAnalysis volumes(device, streams);
  bool ok = true;
  auto traced = traceRoutedStreams(device);
  ok &= require(traced.size() == 2 && traced[0].packetID == 7 &&
                    traced[1].packetID == 9,
                "ControlMM2S tracing used another engine's packet IDs");
  ok &= require(volumes.sendVolume(data) == 64, "data channel volume changed");
  ok &= require(volumes.sendVolume(control) == 128,
                "ControlMM2S volume aliases ordinary MM2S");
  ok &= require(volumes.sendVolume(controlThen) == 32,
                "control packet ID filtering lost");
  ok &= require(volumes.maySendAfter(control, controlThen) == true &&
                    volumes.maySendAfter(controlThen, control) == false,
                "ControlMM2S BD packet order lost");
  ok &= require(volumes.receiveCapacity(data.dst) == 64,
                "TraceS2MM receiver capacity lost");
  ok &= require(volumes.receiveCapacity({{0, 0}, {WireBundle::DMA, 0}}) == 32,
                "trace and data receive capacities alias");
  ok &= require(!volumes.canFill(data.dst, streams),
                "a fitting trace transfer is classified as blocking");
  data.keepsPktHeader = true;
  SmallVector<RoutedStream, 4> withHeader{data};
  ok &= require(volumes.canFill(data.dst, withHeader),
                "trace header overflow is not classified as blocking");
  return ok;
}

// A header contributes to receiver backpressure even when the router never
// inspects it. Exercise actual traced paths rather than constructing streams
// with keepsPktHeader set by the test.
static bool checkCircuitHeaders(MLIRContext &context) {
  const char *routes[] = {"aie.connect<DMA : 0, East : 0>",
                          R"mlir(
        %a = aie.amsel<0> (0)
        aie.masterset(East : 0, %a) {keep_pkt_header = true}
        aie.packet_rules(DMA : 0) { aie.rule(31, 1, %a) }
      )mlir",
                          R"mlir(
        %a = aie.amsel<0> (0)
        aie.masterset(East : 0, %a) {keep_pkt_header = false}
        aie.packet_rules(DMA : 0) { aie.rule(31, 1, %a) }
      )mlir",
                          "aie.connect<DMA : 0, East : 0>"};
  for (unsigned mode = 0; mode < 4; ++mode) {
    bool packetProducer = mode != 3;
    std::string input = R"mlir(
      module {
        aie.device(npu3) {
          %source = aie.tile(0, 0)
          %dest = aie.tile(1, 0)
          %source_sw = aie.switchbox(%source) {
    )mlir";
    input += routes[mode];
    input += R"mlir(
          }
          %dest_sw = aie.switchbox(%dest) {
            aie.connect<West : 0, DMATrace : 0>
          }
          %buffer = aie.external_buffer : memref<64xi32>
          %source_dma = aie.shim_dma(%source) {
            %start = aie.dma_start(MM2S, 0, ^bd, ^end)
          ^bd:
            aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 16)
    )mlir";
    if (packetProducer)
      input += "{packet = #aie.packet_info<pkt_id = 1, pkt_type = 0>}";
    input += R"mlir(
            aie.next_bd ^end
          ^end:
            aie.end
          }
          %dest_dma = aie.shim_dma(%dest) {
            %start = aie.dma_start(TraceS2MM, 0, ^bd, ^end)
          ^bd:
            aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 16)
            aie.next_bd ^end
          ^end:
            aie.end
          }
        }
      }
    )mlir";
    auto module = parseSourceString<ModuleOp>(input, &context);
    if (!module)
      return false;
    auto device = *module->getOps<DeviceOp>().begin();
    auto streams = traceRoutedStreams(device);
    if (!require(streams.size() == 1 &&
                     streams[0].src ==
                         StreamEndpoint{{0, 0}, {WireBundle::DMA, 0}} &&
                     streams[0].dst ==
                         StreamEndpoint{{1, 0}, {WireBundle::DMATrace, 0}},
                 "circuit header test lost its actual DMA-to-trace route"))
      return false;
    StreamVolumeAnalysis volumes(device, streams);
    bool headerArrives = packetProducer && mode != 2;
    if (!require(volumes.sendVolume(streams[0]) == (headerArrives ? 68u : 64u),
                 "tracing a final circuit hop lost or invented packet bytes") ||
        !require(volumes.receiveCapacity(streams[0].dst) == 64,
                 "trace receiver capacity changed in circuit header test") ||
        !require(volumes.canFill(streams[0].dst, streams) == headerArrives,
                 "final circuit hop caused incorrect receiver backpressure"))
      return false;

    if (mode != 0 && mode != 3)
      continue;
    // Check the same all-circuit design before routing: packet production is
    // configured by the DMA BD even though aie.flow carries no packet ID.
    auto tiles = device.getOps<TileOp>();
    auto it = tiles.begin();
    TileOp source = *it++;
    TileOp dest = *it;
    OpBuilder builder(device.getBody()->getTerminator());
    FlowOp::create(builder, device.getLoc(), source, WireBundle::DMA, 0, dest,
                   WireBundle::DMATrace, 0);
    auto requested = requestedStreams(device);
    if (!require(requested.size() == 1,
                 "circuit request was not represented as one stream"))
      return false;
    StreamVolumeAnalysis requestedVolumes(device, requested);
    if (!require(requestedVolumes.sendVolume(requested[0]) ==
                     (packetProducer ? 68u : 64u),
                 "unrouted circuit request lost or invented packet bytes") ||
        !require(requestedVolumes.canFill(requested[0].dst, requested) ==
                     packetProducer,
                 "unrouted circuit request misclassified trace backpressure"))
      return false;
  }
  return true;
}

static bool checkWaits(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      aie.device(npu3) {
        %shim = aie.tile(0, 0)
        aie.runtime_sequence() {
          %z = arith.constant 0 : i32
          %one = arith.constant 1 : i32
          %ctrl = arith.constant 2 : i32
          %trace = arith.constant 3 : i32
          aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %z repeat %z {issue_token = true} : i32, i32
          aiex.npu.push_queue(0, 0, ControlMM2S : 0) bd_id %z repeat %z {issue_token = true} : i32, i32
          aiex.npu.push_queue(0, 0, TraceS2MM : 0) bd_id %z repeat %z {issue_token = true} : i32, i32
          aiex.npu.sync(%z, %z, %ctrl, %z, %one, %one) : i32, i32, i32, i32, i32, i32
          aiex.npu.push_queue(0, 0, S2MM : 0) bd_id %z repeat %z {issue_token = true} : i32, i32
          aiex.npu.sync(%z, %z, %trace, %z, %one, %one) : i32, i32, i32, i32, i32, i32
          aiex.npu.push_queue(0, 0, MM2S : 1) bd_id %z repeat %z {issue_token = true} : i32, i32
        }
      }
    }
  )mlir",
                                            &context);
  if (!module)
    return false;
  auto device = *module->getOps<DeviceOp>().begin();
  StreamVolumeAnalysis volumes(device, {});
  StreamWaitGraph graph(device, {}, volumes);
  auto data = graph.agentAt({{0, 0}, {WireBundle::DMA, 0}}, true);
  auto control = graph.agentAt({{0, 0}, {WireBundle::DMAControl, 0}}, true);
  auto trace = graph.agentAt({{0, 0}, {WireBundle::DMATrace, 0}}, false);
  auto receiver = graph.agentAt({{0, 0}, {WireBundle::DMA, 0}}, false);
  auto later = graph.agentAt({{0, 0}, {WireBundle::DMA, 1}}, true);
  if (!require(data && control && trace && receiver && later,
               "a specialized stream endpoint has no channel agent"))
    return false;
  auto waitsOn = [&](unsigned from, unsigned to) {
    return !graph.waitChain({from}, {to}, {}).empty();
  };
  bool ok = true;
  ok &= require(waitsOn(*receiver, *control) && !waitsOn(*receiver, *data),
                "npu.sync direction 2 aliases ordinary MM2S");
  ok &= require(waitsOn(*later, *trace) && !waitsOn(*later, *data),
                "npu.sync direction 3 aliases ordinary MM2S");
  auto token = StreamWaitGraph::Agent::at(
      device.getTargetModel(), {{0, 0}, {WireBundle::Control32, 0}}, true);
  ok &=
      require(token && token->kind == StreamWaitGraph::Agent::Kind::Controller,
              "Control32 token sender lost controller semantics");
  ok &= require(!StreamWaitGraph::Agent::at(
                    device.getTargetModel(),
                    {{0, 0}, {WireBundle::Controller32, 0}}, false),
                "Controller32 protocol sink incorrectly modeled as DMA");
  return ok;
}

static bool checkTopology(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      aie.device(npu3) {
        %shim = aie.tile(0, 0)
        %other = aie.tile(1, 0)
        %mem = aie.tile(0, 1)
        %ss = aie.switchbox(%shim) {
          aie.connect<DMAControl : 1, NorthControl : 1>
          aie.connect<North32 : 1, Switch512 : 1>
          aie.connect<Control32 : 0, East32 : 0>
          %a = aie.amsel<0> (0)
          aie.masterset(DMATrace : 0, %a) {keep_pkt_header = true}
          aie.packet_rules(Switch32 : 1) {
            aie.rule(31, 3, %a)
          }
        }
        %ms = aie.switchbox(%mem) {
          aie.connect<DMA : 0, Switch32 : 1>
          aie.connect<Switch512 : 1, South32 : 1>
          %a = aie.amsel<0> (0)
          aie.masterset(TileControl : 0, %a)
          aie.packet_rules(SouthControl : 1) {
            aie.rule(31, 26, %a)
          }
        }
        %os = aie.switchbox(%other) {
          %a = aie.amsel<0> (0)
          aie.masterset(Controller32 : 0, %a)
          aie.packet_rules(West32 : 0) {
            aie.rule(31, 0, %a)
          }
        }
      }
    }
  )mlir",
                                            &context);
  if (!module)
    return false;
  auto device = *module->getOps<DeviceOp>().begin();
  auto streams = traceRoutedStreams(device);
  bool ok = require(streams.size() == 3,
                    "fabric inputs were traced as additional stream sources");
  auto find = [&](WireBundle source) -> const RoutedStream * {
    auto it = llvm::find_if(streams, [&](const RoutedStream &stream) {
      return stream.src.port.bundle == source;
    });
    return it == streams.end() ? nullptr : &*it;
  };
  auto data = find(WireBundle::DMA);
  ok &= require(
      data && data->src.tile == TileID{0, 1} &&
          data->dst == StreamEndpoint{{0, 0}, {WireBundle::DMATrace, 0}} &&
          data->packetID == 3 && data->keepsPktHeader &&
          data->hops.size() == 4 &&
          data->hops[1].input == Port{WireBundle::Switch512, 1} &&
          data->hops[2].input == Port{WireBundle::North32, 1} &&
          data->hops[3].input == Port{WireBundle::Switch32, 1},
      "bridge traversal lost endpoint, header, hop or channel identity");
  auto control = find(WireBundle::DMAControl);
  ok &= require(control &&
                    control->dst ==
                        StreamEndpoint{{0, 1}, {WireBundle::TileControl, 0}} &&
                    control->hops.size() == 2 &&
                    control->hops[1].input == Port{WireBundle::SouthControl, 1},
                "one-way control wire ended before the receiving tile");
  auto token = find(WireBundle::Control32);
  ok &= require(token &&
                    token->dst ==
                        StreamEndpoint{{1, 0}, {WireBundle::Controller32, 0}} &&
                    token->hops.size() == 2 &&
                    token->hops[1].input == Port{WireBundle::West32, 0},
                "narrow horizontal token route lost its controller endpoint");
  ok &=
      require(!getLinkedInput({0, 1}, {WireBundle::SouthControl, 1}) &&
                  !getLinkedInput({0, 0}, {WireBundle::DMATrace, 0}),
              "input-only or terminal port gained a fictitious outgoing wire");
  return ok;
}

int main() {
  MLIRContext context;
  context.getOrLoadDialect<AIEDialect>();
  context.getOrLoadDialect<AIEX::AIEXDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  return checkVolumes(context) && checkCircuitHeaders(context) &&
                 checkWaits(context) && checkTopology(context)
             ? 0
             : 1;
}
