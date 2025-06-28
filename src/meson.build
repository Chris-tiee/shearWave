lib_sources = [  # All source files (excluding headers)
    'hello.cpp',
    'constants.cpp',
    'utilities.cpp'

]

lib = static_library(
    'lib',  # Unique name
    lib_sources,
    dependencies : [mpi, kokkoscore]
)

lib_incdirs = include_directories('.')
