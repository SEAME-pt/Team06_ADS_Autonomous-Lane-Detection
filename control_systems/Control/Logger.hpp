#pragma once
#include <cstdarg>
#include <ctime>
#include <cstdio>

#define CONSOLE_COLOR_RESET   "\x1b[0m"
#define CONSOLE_COLOR_RED     "\x1b[31m"
#define CONSOLE_COLOR_GREEN   "\x1b[32m"
#define CONSOLE_COLOR_CYAN    "\x1b[36m"
#define CONSOLE_COLOR_PURPLE  "\x1b[35m"


inline void Log(int severity, const char *fmt, ...)
{
    const char *type;
    const char *color;
    switch (severity)
    {
    case 0:
        type = "info";
        color = CONSOLE_COLOR_GREEN;
        break;
    case 1:
        type = "warning";
        color = CONSOLE_COLOR_PURPLE;
        break;
    case 2:
        type = "error";
        color = CONSOLE_COLOR_RED;
        break;
    default:
        return;
    }

    time_t rawTime;
    struct tm *timeInfo;
    char timeBuffer[80];
    time(&rawTime);
    timeInfo = localtime(&rawTime);
    strftime(timeBuffer, sizeof(timeBuffer), "[%H:%M:%S]", timeInfo);

    char consoleFormat[1024];
    snprintf(consoleFormat, 1024, "%s%s %s%s%s: %s\n",
             CONSOLE_COLOR_CYAN, timeBuffer,
             color, type, CONSOLE_COLOR_RESET, fmt);

    va_list argptr;
    va_start(argptr, fmt);
    vprintf(consoleFormat, argptr);
    va_end(argptr);
}


#define LOG_INFO(fmt, ...)    Log(0, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)    Log(1, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...)   Log(2, fmt, ##__VA_ARGS__)
