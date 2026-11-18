
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "opencv_compat.hpp"

namespace McCalib {

/**
 * @brief Convert a path-like string to std::filesystem::path.
 *
 * @param item_name Input path string.
 * @return Converted filesystem path.
 */
std::filesystem::path convertStrToPath(const std::string &item_name);

/**
 * @brief Convert a vector of path strings to filesystem paths.
 *
 * @param input Input list of path strings.
 * @return Converted list of filesystem paths.
 */
std::vector<std::filesystem::path>
convertVecStrToVecPath(const std::vector<std::string> &input);

/**
 * @brief Convert an ArUco dictionary name (e.g. "DICT_4X4_50") to its OpenCV
 * predefined dictionary id.
 *
 * An empty string returns the default DICT_6X6_1000. Unknown names return -1.
 *
 * @param name Dictionary name, with or without the "DICT_" prefix.
 * @return OpenCV predefined dictionary id, or -1 if unknown.
 */
int arucoDictionaryIdFromName(const std::string &name);

#if MC_CALIB_USE_LEGACY_ARUCO_API

/**
 * @brief Create one OpenCV Charuco board descriptor per board id.
 *
 * @param num_board Number of boards to generate.
 * @param number_x_square_per_board Number of squares along x for each board.
 * @param number_y_square_per_board Number of squares along y for each board.
 * @param length_square Square side length in user units.
 * @param length_marker Marker side length in user units.
 * @param dict ArUco dictionary used by all generated boards.
 * @param legacy_pattern Use the OpenCV < 4.6 square layout (only matters for
 * boards with an even number of rows; always true with the legacy API).
 * @return Map keyed by board id with allocated Charuco board descriptors.
 */
std::map<int, cv::Ptr<cv::aruco::CharucoBoard>>
createCharucoBoards(const unsigned int num_board,
                    const std::vector<int> &number_x_square_per_board,
                    const std::vector<int> &number_y_square_per_board,
                    const float length_square, const float length_marker,
                    const cv::Ptr<cv::aruco::Dictionary> dict,
                    const bool legacy_pattern = true);
#else
/**
 * @brief Create one OpenCV Charuco board descriptor per board id.
 *
 * @param num_board Number of boards to generate.
 * @param number_x_square_per_board Number of squares along x for each board.
 * @param number_y_square_per_board Number of squares along y for each board.
 * @param length_square Square side length in user units.
 * @param length_marker Marker side length in user units.
 * @param dict ArUco dictionary used by all generated boards.
 * @param legacy_pattern Use the OpenCV < 4.6 square layout (only matters for
 * boards with an even number of rows; always true with the legacy API).
 * @return Map keyed by board id with allocated Charuco board descriptors.
 */
std::map<int, cv::Ptr<cv::aruco::CharucoBoard>>
createCharucoBoards(const unsigned int num_board,
                    const std::vector<int> &number_x_square_per_board,
                    const std::vector<int> &number_y_square_per_board,
                    const float length_square, const float length_marker,
                    const cv::aruco::Dictionary &dict,
                    const bool legacy_pattern = false);
#endif

/**
 * @brief Render Charuco board images from per-board geometry and resolution.
 *
 * @param num_board Number of boards to render.
 * @param number_x_square_per_board Number of squares along x for each board.
 * @param number_y_square_per_board Number of squares along y for each board.
 * @param length_square Square side length in user units.
 * @param length_marker Marker side length in user units.
 * @param resolution_x_per_board Output width per board image.
 * @param resolution_y_per_board Output height per board image.
 * @param dictionary_id OpenCV predefined ArUco dictionary id.
 * @param legacy_pattern Use the OpenCV < 4.6 square layout (new API only).
 * @return Vector of rendered board images.
 */
std::vector<cv::Mat> createCharucoBoardsImages(
    const unsigned int num_board,
    const std::vector<int> &number_x_square_per_board,
    const std::vector<int> &number_y_square_per_board,
    const float length_square, const float length_marker,
    const std::vector<int> &resolution_x_per_board,
    const std::vector<int> &resolution_y_per_board,
    const int dictionary_id = cv::aruco::DICT_6X6_1000,
    const bool legacy_pattern = false);

} // namespace McCalib