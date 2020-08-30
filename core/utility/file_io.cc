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
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"

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

class FileReaderSource::ReaderInternal {
public:
    ReaderInternal(tensorflow::RandomAccessFile* p_file)
        : file_(p_file)
        , stream(p_file)
        , buf(&stream, 65535) {
    }

    ~ReaderInternal() {
    }

public:
    std::unique_ptr<tensorflow::RandomAccessFile> file_;

    tensorflow::io::RandomAccessInputStream stream;
    tensorflow::io::BufferedInputStream buf;
};

FileReaderSource::FileReaderSource(const std::string& file) {
    std::unique_ptr<tensorflow::RandomAccessFile> reader;
    CHECK_TF_STATUS(tensorflow::Env::Default()->NewRandomAccessFile(file, &reader));
    reader_ = std::make_shared<ReaderInternal>(reader.release());
}

FileReaderSource::FileReaderSource(const FileReaderSource& reader_source)
    : reader_(reader_source.reader_)
{ }

FileReaderSource::~FileReaderSource() {
}

std::streamsize FileReaderSource::read(char_type* str, std::streamsize n) {
    tensorflow::tstring buffer;
    auto s = reader_->buf.ReadNBytes(n, &buffer);
    CHECK_LE(buffer.size(), n);

    if (buffer.size() == 0 && tensorflow::errors::IsOutOfRange(s)) {
        return -1;
    }

    std::copy(buffer.begin(), buffer.end(), str);

    return buffer.size();
}

} // namespace tensornet

