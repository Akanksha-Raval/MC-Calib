#include "utilities.hpp"

#include <algorithm>
#include <cctype>
#include <numeric>

namespace McCalib {

/**
 * @brief Convert a string path representation to std::filesystem::path.
 *
 * @param item_name Input path string.
 * @return Filesystem path with the same content.
 */
std::filesystem::path convertStrToPath(const std::string &item_name) {
  const std::filesystem::path path(item_name);
  return path;
}

/**
 * @brief Convert a vector of strings into filesystem paths.
 *
 * @param input Input vector of path strings.
 * @return Converted vector of filesystem paths.
 */
std::vector<std::filesystem::path>
convertVecStrToVecPath(const std::vector<std::string> &input) {
  std::vector<std::filesystem::path> out;
  out.reserve(input.size());
  for (const std::string &item : input) {
    const std::filesystem::path path(item);
    out.push_back(path);
  }
  return out;
}

/**
 * @brief Convert an ArUco dictionary name to its OpenCV predefined id.
 *
 * @param name Dictionary name, e.g. "DICT_4X4_50" (prefix "DICT_" optional).
 * @return OpenCV dictionary id, DICT_6X6_1000 if empty, -1 if unknown.
 */
int arucoDictionaryIdFromName(const std::string &name) {
  if (name.empty())
    return cv::aruco::DICT_6X6_1000;
  std::string key = name;
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (key.rfind("DICT_", 0) != 0)
    key = "DICT_" + key;
  static const std::map<std::string, int> dictionaries = {
      {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
      {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
      {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
      {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
      {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
      {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
      {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
      {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
      {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
      {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
      {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
      {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
      {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
      {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
      {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
      {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
      {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
  };
  const auto it = dictionaries.find(key);
  return it == dictionaries.end() ? -1 : it->second;
}

#if MC_CALIB_USE_LEGACY_ARUCO_API

/**
 * @brief Create one Charuco board descriptor per configured board id.
 *
 * Marker ids are offset from one board to the next so multiple boards can be
 * detected in the same dataset without id collisions.
 *
 * @param num_board Number of boards to generate.
 * @param number_x_square_per_board Number of squares along x for each board.
 * @param number_y_square_per_board Number of squares along y for each board.
 * @param length_square Square side length.
 * @param length_marker Marker side length.
 * @param dict ArUco dictionary shared by all boards.
 * @return Map of board ids to allocated Charuco board descriptors.
 */
std::map<int, cv::Ptr<cv::aruco::CharucoBoard>>
createCharucoBoards(const unsigned int num_board,
                    const std::vector<int> &number_x_square_per_board,
                    const std::vector<int> &number_y_square_per_board,
                    const float length_square, const float length_marker,
                    const cv::Ptr<cv::aruco::Dictionary> dict,
                    const bool /*legacy_pattern*/) {
  std::map<int, cv::Ptr<cv::aruco::CharucoBoard>> charuco_boards;
  int offset_count = 0;
  for (std::size_t i = 0; i < num_board; i++) {
    // declare the board
    cv::Ptr<cv::aruco::CharucoBoard> charuco = cv::aruco::CharucoBoard::create(
        number_x_square_per_board[i], number_y_square_per_board[i],
        length_square, length_marker, dict);
    // If it is the first board then just use the standard idx
    if (i != 0) {
      int id_offset = charuco_boards[i - 1]->ids.size() + offset_count;
      offset_count = id_offset;
      for (auto &id : charuco->ids) {
        id += id_offset;
      }
    }
    charuco_boards[i] = charuco;
  }
  assert(charuco_boards.size() == num_board);
  return charuco_boards;
}

/**
 * @brief Render one printable image per configured Charuco board.
 *
 * @param num_board Number of boards to render.
 * @param number_x_square_per_board Number of squares along x for each board.
 * @param number_y_square_per_board Number of squares along y for each board.
 * @param length_square Square side length.
 * @param length_marker Marker side length.
 * @param resolution_x_per_board Output width per board.
 * @param resolution_y_per_board Output height per board.
 * @return Rendered board images.
 */
std::vector<cv::Mat>
createCharucoBoardsImages(const unsigned int num_board,
                          const std::vector<int> &number_x_square_per_board,
                          const std::vector<int> &number_y_square_per_board,
                          const float length_square, const float length_marker,
                          const std::vector<int> &resolution_x_per_board,
                          const std::vector<int> &resolution_y_per_board,
                          const int dictionary_id,
                          const bool legacy_pattern) {
  cv::Ptr<cv::aruco::Dictionary> dict =
      cv::aruco::getPredefinedDictionary(dictionary_id);
  const std::map<int, cv::Ptr<cv::aruco::CharucoBoard>> charuco_boards =
      createCharucoBoards(num_board, number_x_square_per_board,
                          number_y_square_per_board, length_square,
                          length_marker, dict, legacy_pattern);

  std::vector<cv::Mat> charuco_boards_images;
  charuco_boards_images.reserve(num_board);
  for (auto const &[i, charuco_board] : charuco_boards) {
    cv::Mat board_image;
    charuco_board->draw(
        cv::Size(resolution_x_per_board[i], resolution_y_per_board[i]),
        board_image, 10, 1);
    charuco_boards_images.push_back(board_image);
  }
  return charuco_boards_images;
}

#else

/**
 * @brief Create one Charuco board descriptor per configured board id.
 *
 * Marker ids are offset from one board to the next so multiple boards can be
 * detected in the same dataset without id collisions.
 *
 * @param num_board Number of boards to generate.
 * @param number_x_square_per_board Number of squares along x for each board.
 * @param number_y_square_per_board Number of squares along y for each board.
 * @param length_square Square side length.
 * @param length_marker Marker side length.
 * @param dict ArUco dictionary shared by all boards.
 * @return Map of board ids to allocated Charuco board descriptors.
 */
std::map<int, cv::Ptr<cv::aruco::CharucoBoard>>
createCharucoBoards(const unsigned int num_board,
                    const std::vector<int> &number_x_square_per_board,
                    const std::vector<int> &number_y_square_per_board,
                    const float length_square, const float length_marker,
                    const cv::aruco::Dictionary &dict,
                    const bool legacy_pattern) {
  std::map<int, cv::Ptr<cv::aruco::CharucoBoard>> charuco_boards;
  int offset_count = 0;
  for (std::size_t i = 0; i < num_board; i++) {
    if (i == 0) {
      // if it is the first board then just use the standard idx
      charuco_boards[i] = cv::makePtr<cv::aruco::CharucoBoard>(
          cv::aruco::CharucoBoard(cv::Size(number_x_square_per_board[i],
                                           number_y_square_per_board[i]),
                                  length_square, length_marker, dict));
    } else {
      int id_offset = charuco_boards[i - 1]->getIds().size() + offset_count;
      offset_count = id_offset;

      // Build a temporary board to query how many marker IDs are required
      // for the current geometry. This allows mixed board sizes.
      const cv::aruco::CharucoBoard current_board_geometry(
          cv::Size(number_x_square_per_board[i], number_y_square_per_board[i]),
          length_square, length_marker, dict);
      const std::size_t num_idxs = current_board_geometry.getIds().size();
      std::vector<int> cur_ids(num_idxs);
      std::iota(cur_ids.begin(), cur_ids.end(), id_offset);

      charuco_boards[i] = cv::makePtr<cv::aruco::CharucoBoard>(
          cv::aruco::CharucoBoard(cv::Size(number_x_square_per_board[i],
                                           number_y_square_per_board[i]),
                                  length_square, length_marker, dict, cur_ids));
    }
  }
#if MC_CALIB_OPENCV_AT_LEAST(4, 8)
  // OpenCV >= 4.6 changed the square layout of boards with an even number of
  // rows; the legacy flag restores the pre-4.6 layout.
  for (auto &it : charuco_boards) {
    it.second->setLegacyPattern(legacy_pattern);
  }
#else
  (void)legacy_pattern;
#endif
  assert(charuco_boards.size() == num_board);
  return charuco_boards;
}

/**
 * @brief Render one printable image per configured Charuco board.
 *
 * @param num_board Number of boards to render.
 * @param number_x_square_per_board Number of squares along x for each board.
 * @param number_y_square_per_board Number of squares along y for each board.
 * @param length_square Square side length.
 * @param length_marker Marker side length.
 * @param resolution_x_per_board Output width per board.
 * @param resolution_y_per_board Output height per board.
 * @return Rendered board images.
 */
std::vector<cv::Mat>
createCharucoBoardsImages(const unsigned int num_board,
                          const std::vector<int> &number_x_square_per_board,
                          const std::vector<int> &number_y_square_per_board,
                          const float length_square, const float length_marker,
                          const std::vector<int> &resolution_x_per_board,
                          const std::vector<int> &resolution_y_per_board,
                          const int dictionary_id,
                          const bool legacy_pattern) {
  const cv::aruco::Dictionary dict =
      cv::aruco::getPredefinedDictionary(dictionary_id);
  const std::map<int, cv::Ptr<cv::aruco::CharucoBoard>> charuco_boards =
      createCharucoBoards(num_board, number_x_square_per_board,
                          number_y_square_per_board, length_square,
                          length_marker, dict, legacy_pattern);

  std::vector<cv::Mat> charuco_boards_images;
  charuco_boards_images.reserve(num_board);
  for (auto const &[i, charuco_board] : charuco_boards) {
    cv::Mat board_image;
    charuco_board->generateImage(
        cv::Size(resolution_x_per_board[i], resolution_y_per_board[i]),
        board_image, 10, 1);
    charuco_boards_images.push_back(board_image);
  }

  return charuco_boards_images;
}

#endif

} // namespace McCalib