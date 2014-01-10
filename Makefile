program_NAME := tracker
program_C_SRCS := $(wildcard *.c)
program_CXX_SRCS := $(wildcard *.cpp)
program_C_OBJS := ${program_C_SRCS:.c=.o}
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_OBJS := $(program_C_OBJS) $(program_CXX_OBJS)
program_INCLUDE_DIRS := /opt/intel/ipp/include /home/yuncong/opencv/release/include
program_LIBRARY_DIRS := /opt/intel/ipp/lib/intel64 /home/yuncong/opencv/release/lib
program_LIBRARIES := opencv_core opencv_highgui opencv_objdetect opencv_imgproc ippi ippcv gd png z jpeg freetype m pthread boost_system boost_filesystem
#CPPFLAGS += -DDTIME -g -ffast-math -mfpmath=387 -march=core2
CPPFLAGS += -g -O2 -fopenmp -ffast-math -mfpmath=387 -march=core2

CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(program_LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(program_LIBRARIES),-l$(library))
LDFLAGS += -Wl,-rpath /opt/intel/ipp/lib/intel64

.PHONY: all clean distclean

all: $(program_NAME)

$(program_NAME): $(program_OBJS)
	$(CXX) $(CPPFLAGS) $(program_OBJS) $(LDFLAGS) -o $(program_NAME)

clean:
	@- $(RM) $(program_NAME)
	@- $(RM) $(program_OBJS)

distclean: clean
