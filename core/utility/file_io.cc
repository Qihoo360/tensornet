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
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"

using tensorflow::io::ZlibOutputBuffer;
using tensorflow::io::ZlibInputStream;
using tensorflow::io::ZlibCompressionOptions;

namespace tensornet {

#define CHECK_TF_STATUS(status)                             \
    do {                                                    \
        auto s = (status);                                  \
        CHECK(s.ok()) << s;                                 \
    } while (0)

FileWriterSink::FileWriterSink(const std::string& file,
            const FileCompressionType compression_type) {
    size_t found = file.find_last_of("/\\");
    CHECK(found != std::string::npos);
    std::string file_dir = file.substr(0, found);

    ZlibCompressionOptions zlib_options;
    zlib_options = ZlibCompressionOptions::GZIP();

    CHECK_TF_STATUS(tensorflow::Env::Default()->RecursivelyCreateDir(file_dir));

    std::unique_ptr<tensorflow::WritableFile> writer;
    CHECK_TF_STATUS(tensorflow::Env::Default()->NewWritableFile(file, &writer));

    // unique_ptr need tensorflow::WritableFile to a complete type when destruct, but we want
    // hide detail of it, so we use the raw ptr directly. this is not safe enough, user must
    // guarantee that only one thread write at the same time
    writer_ = std::move(writer);

    if (FCT_ZLIB == compression_type) {
        ZlibOutputBuffer* zlib_output_buffer = new ZlibOutputBuffer(
                writer_.get(), zlib_options.input_buffer_size,
                zlib_options.output_buffer_size, zlib_options);
        CHECK_TF_STATUS(zlib_output_buffer->Init());

        zlib_writer_ = std::shared_ptr<tensorflow::WritableFile>(zlib_output_buffer);
    }
}

FileWriterSink::FileWriterSink(const FileWriterSink& writer_sink)
    : writer_(writer_sink.writer_)
    , zlib_writer_(writer_sink.zlib_writer_)
{ }

FileWriterSink::~FileWriterSink() {
    if (zlib_writer_ && zlib_writer_.use_count() == 1) {
        zlib_writer_->Flush();
        zlib_writer_->Close();
    }

    zlib_writer_ = nullptr;
    writer_ = nullptr;
}

std::streamsize FileWriterSink::write(const char_type* str, std::streamsize n) {
    if (zlib_writer_) {
        CHECK_TF_STATUS(zlib_writer_->Append(tensorflow::StringPiece(str, n)));
    } else {
        CHECK_TF_STATUS(writer_->Append(tensorflow::StringPiece(str, n)));
    }
    return n;
}

class FileReaderSource::ReaderInternal {
public:
    ReaderInternal(tensorflow::RandomAccessFile* p_file,
            const FileCompressionType compression_type)
        : file_(p_file)
        , raw_stream(p_file)
        , buffer_input_stream(&raw_stream, 65535) {

        compression_type_ = compression_type;

        ZlibCompressionOptions zlib_options;
        zlib_options = ZlibCompressionOptions::GZIP();

        if (FCT_ZLIB == compression_type_) {
            zlib_input_stream.reset(new ZlibInputStream(
                        &buffer_input_stream, zlib_options.input_buffer_size,
                        zlib_options.output_buffer_size, zlib_options));
        }
    }

    ~ReaderInternal() {
        file_ = nullptr;
        zlib_input_stream = nullptr;
    }

    tensorflow::io::InputStreamInterface* GetInputStream() {
        if (FCT_ZLIB == compression_type_) {
            return zlib_input_stream.get();
        } else {
            return &buffer_input_stream;
        }
    }

public:
    std::unique_ptr<tensorflow::RandomAccessFile> file_;

    tensorflow::io::RandomAccessInputStream raw_stream;
    tensorflow::io::BufferedInputStream buffer_input_stream;
    std::unique_ptr<tensorflow::io::InputStreamInterface> zlib_input_stream;

private:
    FileCompressionType compression_type_;
};

FileReaderSource::FileReaderSource(const std::string& file,
            const FileCompressionType compression_type) {
    std::unique_ptr<tensorflow::RandomAccessFile> reader;
    CHECK_TF_STATUS(tensorflow::Env::Default()->NewRandomAccessFile(file, &reader));
    reader_ = std::make_shared<ReaderInternal>(reader.release(), compression_type);
}


FileReaderSource::~FileReaderSource() {
    reader_ = nullptr;
}

std::streamsize FileReaderSource::read(char_type* str, std::streamsize n) {
    tensorflow::tstring buffer;
    auto s = reader_->GetInputStream()->ReadNBytes(n, &buffer);

    if (buffer.size() > (size_t)n) {
        // can't happen
        return -1;
    }

    if (buffer.size() == 0 && tensorflow::errors::IsOutOfRange(s)) {
        return -1;
    }

    std::copy(buffer.begin(), buffer.end(), str);

    return buffer.size();
}

} // namespace tensornet

