#ifndef Timer_hpp
#define Timer_hpp

#include <stdio.h>
#include <chrono>

using namespace std::chrono;

class Timer
{
public:
    Timer(): m_begin(high_resolution_clock::now()) {};
    
    void reset() {m_begin = high_resolution_clock::now();}
    
    int64_t elapsed() const;
    
    int64_t elapsed_micro() const;
    
    int64_t elapsed_nano() const;
    
    int64_t elapsed_minutes() const;
    
    int64_t elapsed_hours() const;
    
    int64_t elapsed_seconds() const;
    
private:
    time_point<high_resolution_clock> m_begin;
};

#endif /* Timer_hpp */
