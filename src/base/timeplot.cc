#include "timeplot.h"

namespace timeplot_ns
{

int double_to_int(double v)
{
    return static_cast<int>(std::round(v));
}

std::ostream& operator<<(std::ostream& out, box_t const& box)
{
    out << "box_t{label=" << box.text << ", row=" << box.row << " [" << box.start << "," << box.stop
        << "]}";
    return out;
}

tuple<int, int, int, int> xywh(tuple<int, int> const& top_left, tuple<int, int> const& bot_right)
{
    auto const& [x0, y0] = top_left;
    auto const& [x1, y1] = bot_right;
    return {x0, y0, x1 - x0, y1 - y0};
}

std::ostream& operator<<(std::ostream& out, svg_box_t const& box)
{
    // Example:
    //  <rect x="100" y="20" rx="5" ry="5" width="100" height="80"
    //  style="fill:lightblue;stroke:pink;stroke-width:5;opacity:1.0" />
    auto [x, y, w, h] = xywh(box.top_left, box.bot_right);
    auto const& [rx, ry] = box.rxy;
    out << "<rect x=" << quote(x + box.stroke_width) << " y=" << quote(y + box.stroke_width)
        << " rx=" << quote(rx) << " ry=" << quote(ry)
        << " width=" << quote(w - 2 * box.stroke_width)
        << " height=" << quote(h - 2 * box.stroke_width) << endl;
    out << " style=\"fill:" << box.fill_color << ";stroke:" << box.stroke_color
        << ";stroke-width:" << (2 * box.stroke_width) << ";opacity:" << box.opacity << "\" />"
        << endl;
    return out;
}

svg_box_t row_background(int which_row, int width, int row_height, string color)
{
    if (color == "") {
        color = which_row % 2 == 0 ? "lightblue" : "pink";
    }
    return svg_box_t{.top_left = {0, row_height * which_row},
                     .bot_right = {width, row_height * (which_row + 1)},
                     .rxy = {0, 0},
                     .fill_color = color,
                     .stroke_color = color,
                     .stroke_width = 0,
                     .opacity = 1.0};
}

std::ostream& operator<<(std::ostream& out, svg_text_t const& text)
{
    // <text x="0" y="15" fill="red"">some text</text>
    auto const& [x, y, w, h] = xywh(text.top_left, text.bot_right);

    out << "<text x=" << quote(x + w / 2) << " y=" << quote(y + h / 2)
        << " fill=" << quote(text.fill_color) << " dominant-baseline=\"middle\""
        << " text-anchor=\"middle\""
        << ">" << text.text << "</text>";

    return out;
}

tuple<svg_box_t, svg_text_t>
task_box(int which_row, int row_height, int start, int finish, string const& text)
{
    tuple<int, int> tl = {start, row_height * which_row};
    tuple<int, int> br = {finish, row_height * (which_row + 1)};
    svg_box_t       svg_box = {.top_left = tl,
                               .bot_right = br,
                               .rxy = {5, 5},
                               .fill_color = "lightgray",
                               .stroke_color = "darkgray",
                               .stroke_width = 1,
                               .opacity = 1.0};
    svg_text_t      svg_text = {
             .top_left = tl,
             .bot_right = br,
             .text = text,
             .fill_color = "black",
    };
    return {svg_box, svg_text};
}

struct subrow_former_t {
    // return the first subrow that has nothing happening
    // in (beg,end)
    int operator()(double beg, double end)
    {
        for (int i = 0; i != items.size(); ++i) {
            if (insert(beg, end, items[i])) {
                return i;
            }
        }
        items.emplace_back();
        items.back().push_back({beg, end});
        return items.size() - 1;
    }

    int num_subrows() const
    {
        return items.size();
    }

    vector<vector<tuple<double, double>>> items;

private:
    static bool insert(double beg, double end, vector<tuple<double, double>>& is)
    {
        // It'd scale better if 'is' is always sorted...
        // But the svg size will be too big to view before
        // this becomes slow
        for (auto const& i : is) {
            if (intersect({beg, end}, i)) {
                return false;
            }
        }
        is.emplace_back(beg, end);
        return true;
    }

    static bool intersect(tuple<double, double> const& lhs, tuple<double, double> const& rhs)
    {
        auto const& [bl, el] = lhs;
        auto const& [br, er] = rhs;
        double b = std::max(bl, br);
        double e = std::min(el, er);
        return b < e;
    }
};

} // namespace timeplot_ns

void timeplot(std::ostream&                     out,
              vector<timeplot_ns::box_t> const& boxes,
              int                               row_height,
              int                               min_box_width,
              optional<double>                  actual_makespan)
{
    int max_image_width = 20000;

    using namespace timeplot_ns;

    double minboxtime = -1.0;
    int    nrow = 0;
    double makespan = 0.0;
    for (auto const& box : boxes) {
        if (box.stop - box.start == 0.0) {
            continue;
        }
        if (minboxtime < 0) {
            minboxtime = box.stop - box.start;
        } else {
            minboxtime = std::min(minboxtime, box.stop - box.start);
        }
        nrow = std::max(nrow, box.row + 1);
        makespan = std::max(makespan, box.stop);
    }

    if (actual_makespan) {
        if (actual_makespan.value() < makespan) {
            throw std::runtime_error("provided makespan less than boxes values!");
        }
        makespan = actual_makespan.value();
    }

    std::function<int(double)> time_to_int;

    time_to_int = [&](double time) { return double_to_int((time / minboxtime) * min_box_width); };

    if (time_to_int(makespan) > max_image_width) {
        time_to_int = [&](double time) {
            return double_to_int((max_image_width * 1.0) * time / makespan);
        };
    }

    int width = time_to_int(makespan);
    int height = nrow * row_height;

    out << "<svg width=" << quote(width) << " height=" << quote(height) << ">" << endl;

    for (int i = 0; i != nrow; ++i) {
        out << row_background(i, width, row_height);
    }

    for (auto const& [row, start, stop, label] : boxes) {
        if (stop - start == 0.0) {
            continue;
        }
        int istart = time_to_int(start);
        int istop = time_to_int(stop);

        if (istop - istart < 30) {
            continue;
        }

        auto [box, text] = task_box(row, row_height, istart, istop, label);

        out << box;
        out << text;
    }

    out << "</svg>" << endl;
}

void timeplot_with_subrow(std::ostream&                     out,
                          vector<timeplot_ns::box_t> const& boxes,
                          int                               subrow_height,
                          int                               min_box_width,
                          optional<double>                  actual_makespan)
{
    int max_image_width = 20000;

    using namespace timeplot_ns;

    double minboxtime = -1.0;
    int    nrow = 0;
    double makespan = 0.0;
    for (auto const& box : boxes) {
        if (box.stop - box.start == 0.0) {
            continue;
        }
        if (minboxtime < 0) {
            minboxtime = box.stop - box.start;
        } else {
            minboxtime = std::min(minboxtime, box.stop - box.start);
        }
        nrow = std::max(nrow, box.row + 1);
        makespan = std::max(makespan, box.stop);
    }

    if (actual_makespan) {
        if (actual_makespan.value() < makespan) {
            throw std::runtime_error("provided makespan less than boxes values!");
        }
        makespan = actual_makespan.value();
    }

    vector<int> num_subrows;
    vector<int> which_subrow;
    {
        vector<subrow_former_t> formers(nrow);
        for (auto const& box : boxes) {
            which_subrow.push_back(formers[box.row](box.start, box.stop));
        }
        for (auto const& former : formers) {
            num_subrows.push_back(former.num_subrows());
        }
    }
    int nsubrow = std::accumulate(num_subrows.begin(), num_subrows.end(), 0);

    std::function<int(double)> time_to_int;

    time_to_int = [&](double time) { return double_to_int((time / minboxtime) * min_box_width); };

    if (time_to_int(makespan) > max_image_width) {
        time_to_int = [&](double time) {
            return double_to_int((max_image_width * 1.0) * time / makespan);
        };
    }

    int width = time_to_int(makespan);
    int height = nsubrow * subrow_height;

    out << "<svg width=" << quote(width) << " height=" << quote(height) << ">" << endl;

    {
        int i = 0;
        for (int r = 0; r != nrow; ++r) {
            string color = r % 2 == 0 ? "lightblue" : "pink";
            int    nsr = num_subrows[r];
            for (int z = 0; z != nsr; ++z) {
                out << row_background(i, width, subrow_height, color);
                i++;
            }
        }
    }

    vector<int> offsets(num_subrows.size());
    std::exclusive_scan(num_subrows.begin(), num_subrows.end(), offsets.begin(), 0);
    for (int b = 0; b != boxes.size(); ++b) {
        auto const& [row, start, stop, label] = boxes[b];
        int const& row_subrow = which_subrow[b];
        int        subrow = offsets[row] + row_subrow;

        if (stop - start == 0.0) {
            continue;
        }
        int istart = time_to_int(start);
        int istop = time_to_int(stop);

        if (istop - istart < 30) {
            continue;
        }

        auto [box, text] = task_box(subrow, subrow_height, istart, istop, label);

        out << box;
        out << text;
    }

    out << "</svg>" << endl;
}
