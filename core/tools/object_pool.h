#ifndef TENSORNET_CORE_TOOLS_OBJECT_POOL_H_
#define TENSORNET_CORE_TOOLS_OBJECT_POOL_H_

#include <list>
#include <vector>
#include <string>

template <typename Object>
class ObjectPool {
public:
    ObjectPool(size_t unSize, const std::string& path, const std::string doneFile)
        : _un_size(unSize)
        , _path(path)
        , _done_file(doneFile) {
        for (size_t un_idx = 0; un_idx < _un_size; ++ un_idx) {
            _o_pool.push_back(new Object(_path, _done_file));
        }
    }

    ~ObjectPool() {
        typename std::list<Object *>::iterator o_it = _o_pool.begin();
        while (o_it != _o_pool.end()) {
            delete (*o_it);
            ++o_it;
        }
        _un_size = 0;
    }

    Object* GetObject() {
        Object* p_obj = NULL;
        if (0 == _un_size) {
            p_obj = new Object(_path, _done_file);
        } else {
            p_obj = _o_pool.front();
            _o_pool.pop_front();
            --_un_size;
        }
        if (p_obj == NULL) { 
            std::cerr << "get pointer is " << p_obj << std::endl;
        }
        return p_obj;
    }

    Object* GetObject(std::vector<Object*>* vec) {
        Object* p_obj = GetObject();
        if (p_obj != NULL) {
            vec->push_back(p_obj);
        }
        return p_obj;
    }

    void ReturnObject(Object * p_obj) {
        _o_pool.push_back(p_obj);
        ++_un_size;
    }

    size_t PoolSize() {
        return _un_size;
    }

private:
    size_t _un_size;
    std::list<Object *> _o_pool;
    std::string _path;
    std::string _done_file;
};

#endif
