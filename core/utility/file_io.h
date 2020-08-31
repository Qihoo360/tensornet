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

#ifndef TENSORNET_CORE_UTILITY_FILE_IO_H_
#define TENSORNET_CORE_UTILITY_FILE_IO_H_

#include <string>
#include <memory>
#include <iosfwd>                          // streamsize
#include <boost/iostreams/categories.hpp>  // sink_tag, source_tag

namespace tensorflow {
    class WritableFile;
}

namespace tensornet {

enum FileCompressionType {
    FCT_NONE = 0,
    FCT_ZLIB = 1,
};

class FileWriterSink {
public:
    typedef char char_type;
    typedef boost::iostreams::sink_tag category;

    explicit FileWriterSink(const std::string& file,
            const FileCompressionType compression_type=FCT_NONE);
    ~FileWriterSink();

    FileWriterSink(const FileWriterSink& writer_sink);

    std::streamsize write(const char_type* str, std::streamsize n);

private:
    std::shared_ptr<tensorflow::WritableFile> writer_;
    std::shared_ptr<tensorflow::WritableFile> zlib_writer_;
};

class FileReaderSource {
public:
    typedef char char_type;
    typedef boost::iostreams::source_tag category;

    explicit FileReaderSource(const std::string& file,
            const FileCompressionType compression_type=FCT_NONE);

    ~FileReaderSource();

    FileReaderSource(const FileReaderSource& reader_source) = default;

    std::streamsize read(char_type* s, std::streamsize n);

    class ReaderInternal;

private:
    std::shared_ptr<ReaderInternal> reader_;

};

}  // namespace tensornet

#endif  // TENSORNET_UTILITY_SEMAPHORE_H_
