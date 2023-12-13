#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "gmock/gmock.h"         // from @googletest
#include "gtest/gtest.h"         // from @googletest
#include "include/Graph/Graph.h"
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace graph {
namespace {

using ::testing::UnorderedElementsAre;

TEST(LevelSortTest, SimpleGraphLevelSort) {
  // Example graph:
  //       ↗ 2 ↘
  // 0 → 1 → 3 → 4
  //   ↘ → → → ↗
  //
  // Level divisions:
  // 0 | 1 | 2 | 3
  //
  // Level 0: node 0
  // Level 1: node 1
  // Level 2: node 2, 3
  // Level 3: node 4

  Graph<std::string_view, int> graph;
  graph.AddVertex("0", 1);
  graph.AddVertex("1", 1);
  graph.AddVertex("2", 1);
  graph.AddVertex("3", 1);
  graph.AddVertex("4", 1);
  graph.AddEdge("0", "1");
  graph.AddEdge("1", "2");
  graph.AddEdge("1", "3");
  graph.AddEdge("1", "4");
  graph.AddEdge("2", "4");
  graph.AddEdge("3", "4");

  auto level_sorted = graph.SortGraphByLevels();
  EXPECT_TRUE(succeeded(level_sorted));
  std::vector<std::vector<std::string_view>> level_unwrapped =
      level_sorted.value();
  EXPECT_EQ(level_unwrapped.size(), 4);
  EXPECT_THAT(level_unwrapped[0], UnorderedElementsAre("0"));
  EXPECT_THAT(level_unwrapped[1], UnorderedElementsAre("1"));
  EXPECT_THAT(level_unwrapped[2], UnorderedElementsAre("2", "3"));
  EXPECT_THAT(level_unwrapped[3], UnorderedElementsAre("4"));
}

TEST(LevelSortTest, MultiInputGraphLevelSort) {
  // Example graph with multiple inputs:
  // 0 → 5 → 6 → 7 → 8 → 9 → 10
  //     1 ↗    ↑    ↑   ↑
  //         2 ↗     ↑   ↑
  //             3 ↗     ↑
  //                 4 ↗
  //
  // Level divisions:
  // 0 | 1 | 2 | 3 | 4 | 5 | 6
  // Level 0: node 0
  // Level 1: node 1, 5
  // Level 2: node 2, 6
  // Level 3: node 3, 7
  // Level 4: node 4, 8
  // Level 5: node 9
  // Level 6: node 10

  Graph<std::string_view, int> graph;
  graph.AddVertex("0", 1);
  graph.AddVertex("1", 1);
  graph.AddVertex("2", 1);
  graph.AddVertex("3", 1);
  graph.AddVertex("4", 1);
  graph.AddVertex("5", 1);
  graph.AddVertex("6", 1);
  graph.AddVertex("7", 1);
  graph.AddVertex("8", 1);
  graph.AddVertex("9", 1);
  graph.AddVertex("10", 1);
  graph.AddEdge("0", "5");
  graph.AddEdge("1", "6");
  graph.AddEdge("2", "7");
  graph.AddEdge("3", "8");
  graph.AddEdge("4", "9");
  graph.AddEdge("5", "6");
  graph.AddEdge("6", "7");
  graph.AddEdge("7", "8");
  graph.AddEdge("8", "9");
  graph.AddEdge("9", "10");

  auto level_sorted = graph.SortGraphByLevels();
  EXPECT_TRUE(succeeded(level_sorted));
  std::vector<std::vector<std::string_view>> level_unwrapped =
      level_sorted.value();
  EXPECT_EQ(level_unwrapped.size(), 7);
  EXPECT_THAT(level_unwrapped[0], UnorderedElementsAre("0"));
  EXPECT_THAT(level_unwrapped[1], UnorderedElementsAre("5", "1"));
  EXPECT_THAT(level_unwrapped[2], UnorderedElementsAre("2", "6"));
  EXPECT_THAT(level_unwrapped[3], UnorderedElementsAre("3", "7"));
  EXPECT_THAT(level_unwrapped[4], UnorderedElementsAre("4", "8"));
  EXPECT_THAT(level_unwrapped[5], UnorderedElementsAre("9"));
  EXPECT_THAT(level_unwrapped[6], UnorderedElementsAre("10"));
}

TEST(LevelSortTest, MultiOutputGraphLevelSort) {
  // Example graph with multiple outputs:
  // 0 → 1 → 2 → 3 → 4 → 5
  //       ↘   ↘   ↘  ↘  6
  //         ↘   ↘   ↘ → 7
  //           ↘   ↘ → → 8
  //             ↘ → → → 9
  //
  // Level divisions:
  // 0 | 1 | 2 | 3 | 4 | 5
  // Level 0: node 0
  // Level 1: node 1
  // Level 2: node 2
  // Level 3: node 3
  // Level 4: node 4
  // Level 5: node 5, 6, 7, 8, 9

  Graph<std::string_view, int> graph;
  graph.AddVertex("0", 1);
  graph.AddVertex("1", 1);
  graph.AddVertex("2", 1);
  graph.AddVertex("3", 1);
  graph.AddVertex("4", 1);
  graph.AddVertex("5", 1);
  graph.AddVertex("6", 1);
  graph.AddVertex("7", 1);
  graph.AddVertex("8", 1);
  graph.AddVertex("9", 1);
  graph.AddEdge("0", "1");
  graph.AddEdge("1", "2");
  graph.AddEdge("2", "3");
  graph.AddEdge("3", "4");
  graph.AddEdge("4", "5");
  graph.AddEdge("4", "6");
  graph.AddEdge("1", "9");
  graph.AddEdge("2", "8");
  graph.AddEdge("3", "7");

  auto level_sorted = graph.SortGraphByLevels();
  EXPECT_TRUE(succeeded(level_sorted));
  std::vector<std::vector<std::string_view>> level_unwrapped =
      level_sorted.value();
  EXPECT_EQ(level_unwrapped.size(), 6);
  EXPECT_THAT(level_unwrapped[0], UnorderedElementsAre("0"));
  EXPECT_THAT(level_unwrapped[1], UnorderedElementsAre("1"));
  EXPECT_THAT(level_unwrapped[2], UnorderedElementsAre("2"));
  EXPECT_THAT(level_unwrapped[3], UnorderedElementsAre("3"));
  EXPECT_THAT(level_unwrapped[4], UnorderedElementsAre("4"));
  EXPECT_THAT(level_unwrapped[5],
              UnorderedElementsAre("5", "6", "7", "8", "9"));
}

}  // namespace
}  // namespace graph
}  // namespace heir
}  // namespace mlir
