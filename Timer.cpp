#include "Timer.hpp"

int64_t Timer::elapsed() const
{
    return duration_cast<milliseconds>(high_resolution_clock::now() - m_begin).count();
}

int64_t Timer::elapsed_hours() const
{
    return duration_cast<hours>(high_resolution_clock::now() - m_begin).count();
}

int64_t Timer::elapsed_micro() const
{
    return duration_cast<microseconds>(high_resolution_clock::now() - m_begin).count();
}

int64_t Timer::elapsed_nano() const
{
    return duration_cast<nanoseconds>(high_resolution_clock::now() - m_begin).count();
}

int64_t Timer::elapsed_seconds() const
{
    return duration_cast<seconds>(high_resolution_clock::now() - m_begin).count();
}

int64_t Timer::elapsed_minutes() const
{
    return duration_cast<minutes>(high_resolution_clock::now() - m_begin).count();
}