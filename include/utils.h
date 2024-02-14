//
// Created by Rich on 2021/1/25.
//

#ifndef LPA_UTILS_H
#define LPA_UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;

class Timer
{
public:
    typedef decltype(std::chrono::system_clock::now()) time_type;

    Timer() {}
    void start() { m_start = my_clock(); }
    void stop() { m_stop = my_clock(); }
    double elapsed_time() const
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(m_stop - m_start).count() / 1e3;
    }
    double elapsed_second() const
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(m_stop - m_start).count() / 1e6;
    }

private:
    time_type m_start, m_stop;
    time_type my_clock() const
    {
        return std::chrono::system_clock::now();
    }
};

template <typename V, typename E>
int binary_search(E left, E right, const std::vector<V> &offsets, const std::vector<V> &vertices, V target)
{
    int ans = 0;
    if (offsets[vertices[left] + 1] - offsets[vertices[left]] < target)
    {
        ans = -1;
    }
    else
    {
        if (offsets[vertices[right] + 1] - offsets[vertices[right]] > target)
        {
            ans = right;
        }
        else
        {
            while (offsets[vertices[left] + 1] - offsets[vertices[left]] > target &&
                   offsets[vertices[right] + 1] - offsets[vertices[right]] < target)
            {
                if (offsets[vertices[(left + right + 1) / 2] + 1] - offsets[vertices[(left + right + 1) / 2]] > target)
                {
                    left = (left + right + 1) / 2;
                    ans = left;
                }
                else
                {
                    right = (left + right - 1) / 2;
                    ans = right;
                }
            }
        }
        ans++;
    }
    if (ans != -1)
    {
        while (offsets[vertices[ans - 1] + 1] - offsets[vertices[ans - 1]] <= target)
        {
            ans--;
        }
    }
    return ans;
}
template <typename T, typename U>
constexpr inline auto max(const T &x, const U &y) -> decltype(x + y)
{
    return x > y ? x : y;
}

template <typename T, typename U, typename... Args>
constexpr inline auto max(const T &x, const U &y, const Args &...rest) -> decltype(x + y)
{
    return max(max(x, y), rest...);
}

template <typename T, typename U>
inline T divup(T x, U y)
{
    return (x + y - 1) / y;
}

#endif //LPA_UTILS_H
