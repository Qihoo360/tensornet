/**
* Author: Zhang Yansheng (zhangyansheng@360.cn)
* Description: 一个双buffer的热加载字典
*/

#ifndef DualDict_h__
#define DualDict_h__

#include <string>
#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <fcntl.h>

#include <boost/shared_ptr.hpp>

namespace hsd {

template <class DictType>
class DictDeleter
{
public:
    DictDeleter(const std::string& msg)
        : _msg(msg)
    { }

    void operator()(const DictType* ptr)
    {
        delete ptr;
    }

private:
    std::string _msg;
};

// 使用这个类的时候务必注意更新间隔时间
template <class DictType>
class DualDict
{
public:
    typedef boost::shared_ptr<DictType> DictTypePtr;

public:
    DualDict()
        : _b_stop(false)
        , _b_inited(false)
        , _reload_interval_sec(5)
        , _mtime(0)
        , _cur_idx(0)
    {
        _revision[0] = 0;
        _revision[1] = 0;
    }

    ~DualDict()
    {
        _b_stop = true;

        pthread_join(_tid, NULL);
    }

    int Init(const std::string& path,
        const std::string& done_file,
        int reload_interval_sec = 10)
    {
        if (_b_inited)
            return -1;

        _mtime = 0;

        _path = path;
        _done_file = done_file;
        _reload_interval_sec = reload_interval_sec;

        if (0 != Reload())
            std::cerr << "Init load error." << std::endl;

        std::cerr << "before _InitMonitorThread:" << _path << std::endl;

        if (0 != _InitMonitorThread())
            return -1;

        std::cerr << "after _InitMonitorThread:" << _path << std::endl;

        _b_inited = true;

        return 0;
    }

    // MENTION! Reload只可在更新线程当中进行，如果单独再调用这个函数会引起线程安全问题
    int Reload()
    {
        time_t new_mtime = _CheckMtime();
        if (new_mtime <= _mtime)
        {
            return 0;
        }

        int backup_idx = 1 - _cur_idx;

        DictTypePtr dict(new DictType(), DictDeleter<DictType>(_path));

        // 目前只支持传递字典的路径，如需要额外配置，请通过RankConfig单例获取
        if (0 != dict->Init(_path))
        {
            std::cerr << "Reload Init error." << std::endl;
            return -1;
        }

        _dict[backup_idx] = dict;

        uint64_t purge_revision = _revision[backup_idx];
        _revision[backup_idx] = dict->Revision();
        _mtime = new_mtime;
        _cur_idx = backup_idx;

        std::cerr << "revision update to " << _revision[1 - backup_idx] << std::endl;

        return 0;
    }

    const DictTypePtr SelectDict(const uint64_t revision) const
    {
        int backup_idx = 1 - _cur_idx;
        if (0 != revision && revision == _revision[backup_idx])
        {
            return _dict[backup_idx];
        }

        if (_revision[_cur_idx] == 0)
        {
            std::cerr << "current dict is invalid" << std::endl;
            return DictTypePtr();
        }

        // 当请求的revision没找到的时候始终使用最新的
        if (revision != _revision[_cur_idx])
        {
            std::cerr << "revision mismatch" << std::endl;
        }

        return _dict[_cur_idx];
    }

    const DictTypePtr GetPreferDict() const
    {
        std::cerr << "prefer:" << _revision[_cur_idx] << std::endl;
        std::cerr << "backup:" << _revision[1-_cur_idx] << std::endl;
        if (_revision[_cur_idx] == 0)
        {
            int backup_idx = 1 - _cur_idx;

            std::cerr << "current dict is invalid " << std::endl;

            return DictTypePtr();
        }

        return _dict[_cur_idx];
    }

    const DictTypePtr GetBackupDict() const
    {
        int backup_idx = 1 - _cur_idx;

        if (_revision[backup_idx] == 0)
        {
            std::cerr << "backup dict is invalid" << std::endl;
            return DictTypePtr();
        }

        return _dict[backup_idx];
    }

    uint64_t GetRevisionPrefer() const
    {
        return _revision[_cur_idx];
    }

    uint64_t GetRevisionBackup() const
    {
        return _revision[1 - _cur_idx];
    }

    bool IsStop() const
    {
        return _b_stop;
    }

    bool IsInit() const
    {
        return _b_inited;
    }

    int ReloadIntervalSec() const
    {
        return _reload_interval_sec;
    }

private:
    int _InitMonitorThread()
    {
        std::cerr << "before pthread_create" << std::endl;
        if (pthread_create(&_tid, 0, DualDict<DictType>::_reload_thread, this) < 0)
        {
            std::cerr << "pthread_create error." << std::endl;
            return -1;
        }
        std::cerr << "after pthread_create" << std::endl;

        return 0;
    }

    time_t _CheckMtime() const
    {
        int fd = open(_done_file.c_str(), O_RDONLY);
        if (fd < 0)
        {
            std::cerr << "open file fail" << std::endl;
            return 0;
        }

        struct stat st;
        if (fstat(fd, &st) != 0)
        {
            std::cerr << "stat file fail" << std::endl;
            close(fd);
            return 0;
        }

        close(fd);

        return st.st_mtime;
    }

    uint64_t _NowUsec() const
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);

        return tv.tv_sec * 1000000ll + tv.tv_usec;
    }

    static void* _reload_thread(void *arg);

private:
    bool _b_stop;
    bool _b_inited;
    int _reload_interval_sec;
    time_t _mtime;
    volatile int _cur_idx;

    uint64_t _revision[2];
    pthread_t _tid;

    DictTypePtr _dict[2];

    std::string _done_file;
    std::string _path;
};

template <class DictType>
void* DualDict<DictType>::_reload_thread(void *arg)
{
    DualDict<DictType>* dual_dict = (DualDict<DictType>*) arg;

    assert(NULL != dual_dict);

    std::cerr << "before _reload_thread" << std::endl;
    int reload_interval_sec = dual_dict->ReloadIntervalSec();
    while (false == dual_dict->IsStop())
    {
        if (0 != dual_dict->Reload())
            std::cerr << "Reload error." << std::endl;

        sleep(reload_interval_sec);
    }
    std::cerr << "after _reload_thread" << std::endl;

    return NULL;
}

} // namespace hsd

#endif // DualDict_h__
