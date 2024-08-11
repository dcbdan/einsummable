#include "timetracker.h"

timetracker_t _global_timetracker;

timetracker_t& get_timetracker()
{
    return _global_timetracker;
}

timetracker_t::timetracker_t() : world_start(now()) {}

timetracker_t::gremlin_t::gremlin_t(timetracker_t& s,
                                    string         name_,
                                    bool           is_i,
                                    bool           is_t,
                                    uint64_t       ff)
    : self(s), name(name_), is_interval(is_i), is_total(is_t), flops(ff), start(s.now())
{
}

timetracker_t::gremlin_t::~gremlin_t()
{
    auto end = now();
    if (is_interval && is_total) {
        self.insert_full(name, start, end, flops);
    } else if (is_interval) {
        self.insert_interval(name, start, end, flops);
    } else if (is_total) {
        self.insert_total(name, start, end, flops);
    }
}

timetracker_t::timestamp_t timetracker_t::now()
{
    return std::chrono::steady_clock::now();
}

void timetracker_t::insert_interval(string                     name,
                                    timetracker_t::timestamp_t start,
                                    timetracker_t::timestamp_t end,
                                    uint64_t                   f)
{
    std::unique_lock lk(m);
    counts[name]++;
    flops[name] += f;
    _insert_interval(name, start, end);
}
void timetracker_t::_insert_interval(string                     name,
                                     timetracker_t::timestamp_t start,
                                     timetracker_t::timestamp_t end)
{
    intervals[name].emplace_back(start, end);
}

void timetracker_t::_insert_total(string                     name,
                                  timetracker_t::timestamp_t start,
                                  timetracker_t::timestamp_t end)
{
    totals[name] += duration_t(end - start);
}
void timetracker_t::insert_total(string                     name,
                                 timetracker_t::timestamp_t start,
                                 timetracker_t::timestamp_t end,
                                 uint64_t                   f)
{
    std::unique_lock lk(m);
    counts[name]++;
    flops[name] += f;
    _insert_total(name, start, end);
}

void timetracker_t::insert_full(string                     name,
                                timetracker_t::timestamp_t start,
                                timetracker_t::timestamp_t end,
                                uint64_t                   f)
{
    std::unique_lock lk(m);
    counts[name]++;
    flops[name] += f;
    _insert_interval(name, start, end);
    _insert_total(name, start, end);
}

timetracker_t::gremlin_t timetracker_t::make_interval_gremlin(string name, uint64_t f)
{
    return gremlin_t(*this, name, true, false, f);
}

timetracker_t::gremlin_t timetracker_t::make_totals_gremlin(string name, uint64_t f)
{
    return gremlin_t(*this, name, false, true, f);
}

timetracker_t::gremlin_t timetracker_t::make_full_gremlin(string name, uint64_t f)
{
    return gremlin_t(*this, name, true, true, f);
}

template <typename V>
int _get_max_msg_size(map<string, V> const& m)
{
    int msg_size = 0;
    for (auto const& [key, _] : m) {
        msg_size = std::max(msg_size, int(key.size()));
    }
    return msg_size;
}

string _fix_msg(string const& msg, int max_msg_size)
{
    return msg + string(2 + (max_msg_size - msg.size()), ' ');
}

void timetracker_t::print_intervals(std::ostream& out)
{
    int max_msg_size = _get_max_msg_size(intervals);
    for (auto const& [key_, intervals] : intervals) {
        string key = _fix_msg(key_, max_msg_size);
        for (auto const& [beg_, end_] : intervals) {
            duration_t beg = beg_ - world_start;
            duration_t end = end_ - world_start;

            out << key << " " << beg.count() << " " << end.count();
        }
    }
}

void timetracker_t::print_totals(std::ostream& out)
{
    int max_msg_size = _get_max_msg_size(totals);
    for (auto const& [key_, duration] : totals) {
        string      key = _fix_msg(key_, max_msg_size);
        string      dur = _fix_msg(write_with_ss(duration.count()), 15);
        string      cnt = _fix_msg(write_with_ss(counts.at(key_)), 10);
        auto const& f = flops.at(key_);
        out << key << " " << dur << " " << cnt << " " << f << std::endl;
    }
}

void timetracker_t::clear()
{
    std::unique_lock lk(m);
    totals.clear();
    intervals.clear();
}
