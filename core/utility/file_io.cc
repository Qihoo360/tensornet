// Copyright (c) 2020, Qihoo, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "core/utility/file_io.h"

#include <string>

#include "tensorflow/core/platform/env.h"

#include <butil/logging.h>

namespace tensornet {

#define CHECK_TF_STATUS(status)                             \
    do {                                                    \
        auto s = (status);                                  \
        CHECK(s.ok()) << s;                                 \
    } while (0)

FileWriterSink::FileWriterSink(const std::string& file) {
    size_t found = file.find_last_of("/\\");
    CHECK(found != std::string::npos);
    std::string file_dir = file.substr(0, found);

    CHECK_TF_STATUS(tensorflow::Env::Default()->RecursivelyCreateDir(file_dir));

    std::unique_ptr<tensorflow::WritableFile> writer;
    CHECK_TF_STATUS(tensorflow::Env::Default()->NewWritableFile(file, &writer));

    // unique_ptr need tensorflow::WritableFile to a complete type when destruct, but we want
    // hide detail of it, so we use the raw ptr directly. this is not safe enough, user must
    // guarantee that only one thread write at the same time
    writer_ = std::move(writer);
}

FileWriterSink::FileWriterSink(const FileWriterSink& writer_sink)
    : writer_(writer_sink.writer_)
{ }

FileWriterSink::~FileWriterSink() {
}

std::streamsize FileWriterSink::write(const char_type* str, std::streamsize n) {
    CHECK_TF_STATUS(writer_->Append(tensorflow::StringPiece(str, n)));
    return n;
}

bool write_to_file(const std::string& file, butil::IOBuf& buf) {
    size_t found = file.find_last_of("/\\");
    CHECK(found != std::string::npos);
    std::string file_dir = file.substr(0, found);

    CHECK_TF_STATUS(tensorflow::Env::Default()->RecursivelyCreateDir(file_dir));

    std::unique_ptr<tensorflow::WritableFile> writer;
    CHECK_TF_STATUS(tensorflow::Env::Default()->NewWritableFile(file, &writer));

    size_t block_num = buf.backing_block_num();
    for (size_t i = 0; i < block_num; ++i) {
        auto piece = buf.backing_block(i);
        CHECK_TF_STATUS(writer->Append(tensorflow::StringPiece(piece.data(), piece.size())));
    }

    return true;
}

bool read_from_file(const std::string& file, butil::IOBuf& buf) {
    std::unique_ptr<tensorflow::RandomAccessFile> reader;
    tensorflow::Env::Default()->NewRandomAccessFile(file, &reader);

    size_t offset = 0;
    tensorflow::StringPiece result;
    const size_t buffer_size = 8 * 1024 * 1024;

    std::unique_ptr<char[]> buffer(new char[buffer_size]);
    char* raw_data = (char*)buffer.get();
    while (true) {
        auto s = reader->Read(offset, buffer_size, &result, raw_data);
        if (s.ok()) {
        } else if (tensorflow::errors::IsOutOfRange(s)) {
            //LOG(INFO) << "Read file EOF: " << file;
        } else {
            LOG(ERROR) << "Read file: " << file << " Error:" << s;
            break;
        }

        if (result.size() == 0) {
            break;
        }
        offset += result.size();
        buf.append(raw_data, result.size());
    }
    return true;
}

} // namespace tensornet

