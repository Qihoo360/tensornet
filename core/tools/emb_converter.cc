#include <queue>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "core/utility/file_io.h"

#include "tensorflow/core/platform/env.h"

#define CHECK_TF_STATUS(status)                             \
    do {                                                    \
        auto s = (status);                                  \
        CHECK(s.ok()) << s;                                 \
    } while (0)

void GetAllFiles(const std::string& path, std::vector<std::string>& files) {
    CHECK_TF_STATUS(tensorflow::Env::Default()->FileExists(path));

    std::queue<std::string> cur;
    cur.emplace(path);

    while (!cur.empty()) {
        std::vector<std::string> childs;
        CHECK_TF_STATUS(tensorflow::Env::Default()->GetChildren(cur.front(), &childs));
        for (auto& item : childs) {
            std::string child_path = cur.front() + "/" + item;
            tensorflow::Status status = tensorflow::Env::Default()->IsDirectory(child_path);
            if (status.ok()) {
                cur.emplace(child_path);
            } else {
                files.emplace_back(child_path);
            }
        }
        cur.pop();
    }
    std::sort(files.begin(), files.end());
}

int ParseAdaGradParams(const std::string& file, butil::IOBuf& buf) {
    butil::IOBuf buf_in;
    if (tensornet::read_from_file(file, buf_in) < 0) {
        LOG(ERROR) << "read_from_file [" << file << "] failed.";
        return -1;
    }

    int dim = 0;

    if (buf_in.size() == 0) {
        LOG(INFO) << "file [" << file << "] processed.";
        return 0;
    }

    CHECK_EQ(sizeof(int), buf_in.cutn(&dim, sizeof(dim)));

    float weight[dim];

    std::vector<std::string> vec;
    boost::split(vec, file, boost::is_any_of("/"));
    std::string table_handle = vec[vec.size() - 3];

    while (buf_in.size()) {
        CHECK(buf_in.size() > sizeof(uint64_t));
        uint64_t key;
        CHECK_EQ(sizeof(uint64_t), buf_in.cutn(&key, sizeof(key)));
        buf.append(std::to_string(key) + "\t" + table_handle);

        size_t no_use_data_len = sizeof(float) + sizeof(uint32_t) + sizeof(int);
        CHECK(buf_in.size() >= dim * sizeof(float) + no_use_data_len);

        CHECK_EQ(sizeof(float) * dim, buf_in.cutn(weight, sizeof(float) * dim));
        CHECK_EQ(no_use_data_len, buf_in.pop_front(no_use_data_len));

        for (int i = 0; i < dim; ++i) {
            buf.append("\t" + std::to_string(weight[i]));
        }

        buf.push_back('\n');
    }
    LOG(INFO) << "file [" << file << "] processed.";

    return 0;
}

int ParseAdamParams(const std::string& file, butil::IOBuf& buf) {
    // TODO (yaolei)

    return 0;
}

int Convert(const std::vector<std::string>& files, const std::string& out_file, const std::string& parse_mode) {
    butil::IOBuf buf;

    for (auto& file : files) {
        if (parse_mode.compare("AdaGrad") == 0) {
            if (ParseAdaGradParams(file, buf) < 0) {
                LOG(ERROR) << "ParseAdaGradParams [" << file << "] Failed.";
                return -1;
            }
        } else if (parse_mode.compare("Adam") == 0) {
            if (ParseAdamParams(file, buf) < 0) {
                LOG(ERROR) << "ParseAdaGradParams [" << file << "] Failed.";
                return -1;
            }
        }
    }

    if (tensornet::write_to_file(out_file, buf) < 0) {
        LOG(ERROR) << "Write data Failed.";
        return -1;
    } 

    return 0;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        LOG(ERROR) << "Wrong Command.\nUsage: " << argv[0] << " [input_path] [output_file] [AdaGrad or Adam]";
        return -1;
    }

    std::string input_path = argv[1];
    std::string out_file = argv[2];
    std::string parse_mode = argv[3];

    std::vector<std::string> files;

    GetAllFiles(input_path, files);

    if (Convert(files, out_file, parse_mode) < 0) {
        LOG(ERROR) << "Convert Failed.";
        return -1;
    }

    return 0;
}
