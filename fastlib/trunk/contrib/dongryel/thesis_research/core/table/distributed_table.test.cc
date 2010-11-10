/** @file distributed_table_test.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */
#include "core/metric_kernels/lmetric.h"
#include "core/table/distributed_table.h"
#include "core/table/mailbox.h"
#include "core/tree/gen_kdtree.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

typedef core::tree::GeneralBinarySpaceTree < core::tree::GenKdTree > TreeType;
typedef core::table::Table<TreeType> TableType;

bool CheckDistributedTableIntegrity(
  const core::table::DistributedTable &table_in,
  const boost::mpi::communicator &world) {
  for(int i = 0; i < world.size(); i++) {
    printf("Process %d thinks Process %d owns %d points.\n",
           world.rank(), i, table_in.local_n_entries(i));
  }
  return true;
}

core::table::DistributedTable *InitDistributedTable(
  boost::mpi::communicator &world,
  boost::mpi::communicator &table_outbox_group,
  boost::mpi::communicator &table_inbox_group) {

  std::pair< core::table::DistributedTable *, std::size_t >
  distributed_table_pair =
    core::table::global_m_file_->UniqueFind<core::table::DistributedTable>();
  core::table::DistributedTable *distributed_table =
    distributed_table_pair.first;

  if(distributed_table == NULL) {
    printf("Process %d: TableOutbox.\n", world.rank());

    // Each process generates its own random data, dumps it to the file,
    // and read its own file back into its own distributed table.
    core::table::Table<TreeType> random_dataset;
    const int num_dimensions = 5;
    int num_points = core::math::RandInt(10, 20);
    random_dataset.Init(5, num_points);
    for(int j = 0; j < num_points; j++) {
      core::table::DensePoint point;
      random_dataset.get(j, &point);
      for(int i = 0; i < num_dimensions; i++) {
        point[i] = core::math::Random(0.1, 1.0);
      }
    }
    printf("Process %d generated %d points...\n", world.rank(), num_points);
    std::stringstream file_name_sstr;
    file_name_sstr << "random_dataset_" << world.rank() << ".csv";
    std::string file_name = file_name_sstr.str();
    random_dataset.Save(file_name);

    std::stringstream distributed_table_name_sstr;
    distributed_table_name_sstr << "distributed_table_" << world.rank() << "\n";
    distributed_table = core::table::global_m_file_->UniqueConstruct <
                        core::table::DistributedTable > ();
    distributed_table->Init(
      file_name, world, table_outbox_group, table_inbox_group);
    printf(
      "Process %d read in %d points...\n",
      world.rank(), distributed_table->local_n_entries());
  }
  return distributed_table;
}

void TableOutboxProcess(
  boost::mpi::communicator &world,
  boost::mpi::communicator &table_outbox_group,
  boost::mpi::communicator &table_inbox_group) {
}

void TableInboxProcess(
  int n_attributes_in,
  boost::mpi::communicator &world,
  boost::mpi::communicator &table_outbox_group,
  boost::mpi::communicator &table_inbox_group) {
  printf("Process %d: TableInbox.\n", world.rank());

  core::table::TableInbox<TableType> inbox;
  inbox.Init(n_attributes_in, &world, &table_outbox_group, &table_inbox_group);
  //inbox.Run();
}

void ComputationProcess(boost::mpi::communicator &world) {
  printf("Process %d: Computation.\n", world.rank());

}

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  if(world.size() <= 1 || world.size() % 3 != 0) {
    std::cout << "Please specify a process number greater than 1 and "
              "a multiple of 3.\n";
    return 0;
  }

  // Initialize the memory allocator.
  core::table::global_m_file_ = new core::table::MemoryMappedFile();
  core::table::global_m_file_->Init(
    std::string("tmp_file"), world.rank(),
    world.rank() % (world.size() / 3), 5000000);

  // Seed the random number.
  srand(time(NULL) + world.rank());

  if(world.rank() == 0) {
    printf("%d processes are present...\n", world.size());
  }

  // If the process ID is less than half of the size of the
  // communicator, make it a table process. Otherwise, make it a
  // computation process. This assignment depends heavily on the
  // round-robin assignment of mpirun.
  boost::mpi::communicator table_outbox_group = world.split(
        (world.rank() < world.size() / 3) ? 1 : 0);
  boost::mpi::communicator table_inbox_group = world.split(
        (world.rank() >= world.size() / 3 &&
         world.rank() < world.size() / 3 * 2) ? 1 : 0);

  core::table::DistributedTable *distributed_table = NULL;

  // Wait until the memory allocator is in synch.
  world.barrier();

  if(world.rank() < world.size() / 3) {
    distributed_table =
      InitDistributedTable(world, table_outbox_group, table_inbox_group);
  }

  // Wait until the distributed table is in synch.
  world.barrier();

  std::pair< core::table::DistributedTable *, std::size_t >
  distributed_table_pair =
    core::table::global_m_file_->UniqueFind<core::table::DistributedTable>();
  distributed_table = distributed_table_pair.first;

  world.barrier();

  printf("Hey from %d: Checking: %d\n", world.rank(),
         distributed_table->n_attributes());

  // The main computation loop.
  if(world.rank() < world.size() / 3) {
    TableOutboxProcess(world, table_outbox_group, table_inbox_group);
  }
  else if(world.rank() < world.size() / 3 * 2) {
    TableInboxProcess(
      distributed_table->n_attributes(), world, table_outbox_group,
      table_inbox_group);
  }
  else {
    ComputationProcess(world);
  }

  return 0;
}
