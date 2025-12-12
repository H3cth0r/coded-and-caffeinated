#include <iostream>
#include <functional>
#include <cmath>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <unistd.h>
#include "riemann.hpp"

class Plotter {
  private:
    Display* display = nullptr;
    int screen;
    Window window;
    GC gc;
    int width=800, height=600;


    bool createWindow() {
      display = XOpenDisplay(NULL);
      if (!display) {
        std::cerr << "Cannot Open display\n";
        return 0;
      }

      screen = DefaultScreen(display);

      window = XCreateSimpleWindow(
          display,
          RootWindow(display, screen),
          0, 0, width, height,
          1,
          BlackPixel(display, screen),
          BlackPixel(display, screen)
      );

      XStoreName(display, window, "Linear Function Plot");
      XSelectInput(display, window, ExposureMask | KeyPressMask);
      XMapWindow(display, window);

      gc = XCreateGC(display, window, 0, NULL);
      return 1;
    }

    std::pair<int, int> world_to_screen(double x, double y) {
        int sx = static_cast<int>(width / 2 + (x + pan_x) * zoom);
        int sy = static_cast<int>(height / 2 - (y + pan_y) * zoom);
        return {sx, sy};
    }

  public:
    std::function<double(double)> evaluateFunction;
    XEvent event;

    double zoom = 200.0;
    double pan_x = 0.0;
    double pan_y = 0.0;

    double x_min = -0.999;
    double x_max =  0.999;
    double bins_num = 30;

    RiemannResults<double> riemannResults;

    Plotter(){
      this->createWindow();
    }
    Plotter(std::function<double(double)> op_t): evaluateFunction(op_t) {
      this->createWindow();
      riemannResults = riemann_sum<double>(op_t, bins_num, x_min, x_max, RiemannRule::Left);
    }

    void addPlot(std::function<double(double)> op_t) {
    }

    
     void plot() {
        XNextEvent(display, &event);

        if (event.type == KeyPress) {
          KeySym key = XLookupKeysym(&event.xkey, 0);

          if (key == XK_Left)      pan_x += 0.1;
          if (key == XK_Right)     pan_x -= 0.1;
          if (key == XK_Up)        pan_y -= 0.1;
          if (key == XK_Down)      pan_y += 0.1;

          if (key == XK_plus || key == XK_equal) zoom *= 1.1;
          if (key == XK_minus)                   zoom /= 1.1;

          XClearWindow(display, window);

          XEvent ev;
          ev.type = Expose;
          ev.xexpose.window = window;
          ev.xexpose.x = 0;
          ev.xexpose.y = 0;
          ev.xexpose.width = width;
          ev.xexpose.height = height;
          ev.xexpose.count = 0;

          XSendEvent(display, window, False, ExposureMask, &ev);
          XFlush(display);

          return;
        }

        if (event.type == Expose) {
          XSetForeground(display, gc, 0xFF0000);
          auto [x0, y0] = world_to_screen(0, 0);
          XDrawLine(display, window, gc, x0, 0, x0, height);

          XSetForeground(display, gc, 0x0000FF);
          XDrawLine(display, window, gc, 0, y0, width, y0);

          XSetForeground(display, gc, 0x00FF00);

          double step = 0.001;

          bool first = true;
          int last_x, last_y;

          for (double x = x_min; x < x_max; x += step) {
            double y;
            if (1 - x * x <= 0) continue;
            y = evaluateFunction(x);

            auto [sx, sy] = world_to_screen(x, y);

            if (!first) {
              XDrawLine(display, window, gc, last_x, last_y, sx, sy);
            }
            first = false;
            last_x = sx;
            last_y = sy;
          } 
        }
     }

     void plotArea() {
        for (auto rect: riemannResults.rects) {
          auto [sx, sy] = world_to_screen(
              rect.x, rect.y
          );
          auto [wx, hy] = world_to_screen(
              rect.width, rect.height
          );
          std::cout << "width: " << rect.width << "\theight: " << rect.height << "\nwx: " << wx << "\thy: " << hy << "\n";
          XFillRectangle(
              display,
              window,
              gc,
              sx, sy,
              (rect.width*zoom), (rect.height*zoom)
          );
        }
     }

     void close() {
       XCloseDisplay(display);
     }

};

int main() {
  auto functionOne = [](double x) {
    return 1/std::sqrt(1 - std::pow(x, 2));
  };

  Plotter plotter = Plotter(functionOne);
  double area = plotter.riemannResults.area;
  std::cout << "area: " << area << "\n";
  
  while (true) {
    plotter.plot();
    plotter.plotArea();
    if (plotter.event.type == KeyPress) {
      KeySym key = XLookupKeysym(&plotter.event.xkey, 0);
      if (key == XK_Escape || key == XK_q) {
        break;
      }
    }
  }

  plotter.close();

  return 0;
}
