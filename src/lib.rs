#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

//! # chili. Rust port of [Spice], a low-overhead parallelization library
//!
//! A crate for very low-overhead fork-join workloads that can potentially be
//! run in parallel.
//!
//! It works best in cases where there are many small computations and where it
//! is expensive to estimate how many are left on the current branch in order
//! to stop trying to share work across threads.
//!
//! [Spice]: https://github.com/judofyr/spice
//!
//! # Examples
//!
//! ```
//! # use chili::{Scope, ThreadPool};
//! struct Node {
//!     val: u64,
//!     left: Option<Box<Node>>,
//!     right: Option<Box<Node>>,
//! }
//!
//! impl Node {
//!     pub fn tree(layers: usize) -> Self {
//!         Self {
//!             val: 1,
//!             left: (layers != 1).then(|| Box::new(Self::tree(layers - 1))),
//!             right: (layers != 1).then(|| Box::new(Self::tree(layers - 1))),
//!         }
//!     }
//! }
//!
//! fn sum<'s, 'a: 's>(node: &'a Node, scope: &mut Scope<'s>) -> u64 {
//!     let (left, right) = scope.join(
//!         |s| node.left.as_deref().map(|n| sum(n, s)).unwrap_or_default(),
//!         |s| node.right.as_deref().map(|n| sum(n, s)).unwrap_or_default(),
//!     );
//!
//!     node.val + left + right
//! }
//!
//! let tree = Node::tree(10);
//! let result: u64;
//!{
//!    let thread_pool = ThreadPool::new();
//!    let mut scope = thread_pool.scope();
//! result = sum(&tree, &mut scope);
//! }
//!
//! assert_eq!(result, 1023);
//! ```

use std::{
    cell::Cell,
    collections::{btree_map::Entry, BTreeMap, HashMap},
    num::NonZero,
    panic,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Barrier, Condvar, Mutex, OnceLock, Weak,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

mod job;

use job::{ExecuteJob, Future, Job, JobQueue, JobStack};

#[derive(Debug)]
struct Heartbeat {
    is_set: Weak<AtomicBool>,
    last_heartbeat: Cell<Instant>,
}

#[derive(Debug, Default)]
struct LockContext {
    time: u64,
    is_stopping: bool,
    heartbeats: HashMap<u64, Heartbeat>,
    heartbeat_index: u64,
}

impl LockContext {
    pub fn new_heartbeat(&mut self) -> Arc<AtomicBool> {
        let is_set = Arc::new(AtomicBool::new(true));
        let heartbeat = Heartbeat {
            is_set: Arc::downgrade(&is_set),
            last_heartbeat: Cell::new(Instant::now()),
        };

        let i = self.heartbeat_index;
        self.heartbeats.insert(i, heartbeat);

        self.heartbeat_index = i.checked_add(1).unwrap();

        is_set
    }
}

#[derive(Debug)]
struct Context {
    lock: Mutex<LockContext>,
    job_is_ready: Condvar,
    scope_created_from_thread_pool: Condvar,
}

fn execute_worker(context: Arc<Context>, barrier: Arc<Barrier>) -> Option<()> {
    let mut first_run = true;

    let mut scope = Scope::new_from_worker(context.clone());

    loop {
        if let Some(job) = scope.pop_earliest_shared_job() {
            // SAFETY:
            // Any `Job` that was shared between threads is waited upon before
            // the `JobStack` exits scope.
            unsafe {
                job.execute(&mut scope);
            }
        }

        if first_run {
            first_run = false;
            barrier.wait();
        };

        let lock = context.lock.lock().ok()?;
        if lock.is_stopping || context.job_is_ready.wait(lock).is_err() {
            break;
        }
    }

    Some(())
}

fn execute_heartbeat(
    context: Arc<Context>,
    heartbeat_interval: Duration,
    num_workers: usize,
) -> Option<()> {
    loop {
        let interval_between_workers = {
            let mut lock = context.lock.lock().ok()?;

            if lock.is_stopping {
                break;
            }

            if lock.heartbeats.len() == num_workers {
                lock = context
                    .scope_created_from_thread_pool
                    .wait_while(lock, |l| l.heartbeats.len() > num_workers)
                    .ok()?;
            }

            let now = Instant::now();
            lock.heartbeats.retain(|_, h| {
                h.is_set
                    .upgrade()
                    .inspect(|is_set| {
                        if now.duration_since(h.last_heartbeat.get()) >= heartbeat_interval {
                            is_set.store(true, Ordering::Relaxed);
                            h.last_heartbeat.set(now);
                        }
                    })
                    .is_some()
            });

            heartbeat_interval.checked_div(lock.heartbeats.len() as u32)
        };

        // If there are no heartbeats (`lock.heartbeats.len()` is 0), skip
        // immediately to the next iteration of the loop to trigger the wait.
        if let Some(interval_between_workers) = interval_between_workers {
            thread::sleep(interval_between_workers);
        }
    }

    Some(())
}

/// A `Scope`d object that you can run fork-join workloads on.
///
/// # Examples
///
/// ```
/// # use chili::ThreadPool;
///
/// let mut vals = [0; 2];
/// let (left, right) = vals.split_at_mut(1);
/// {
/// let mut tp = ThreadPool::new();
/// let mut s = tp.scope();
/// s.join(|_| left[0] = 1, |_| right[0] = 1);
/// }
/// assert_eq!(vals, [1; 2]);
/// ```

pub struct Scope<'s> {
    context: Arc<Context>,
    job_queue: JobQueue<'s>,
    heartbeat: Arc<AtomicBool>,
    join_count: u8,
    shared_jobs: Mutex<BTreeMap<usize, (u64, Box<dyn ExecuteJob<'s> + 's>)>>,
}

impl<'s> Scope<'s> {
    /// Returns the global scope.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chili::Scope;
    /// let _s = Scope::global();
    /// ```
    pub fn global() -> Scope<'static> {
        ThreadPool::global().scope()
    }

    fn new_from_thread_pool(thread_pool: &'s ThreadPool) -> Self {
        let heartbeat = thread_pool.context.lock.lock().unwrap().new_heartbeat();
        thread_pool
            .context
            .scope_created_from_thread_pool
            .notify_one();

        Self {
            context: thread_pool.context.clone(),
            job_queue: JobQueue::default(),
            heartbeat,
            join_count: 0,
            shared_jobs: Default::default(),
        }
    }

    fn new_from_worker(context: Arc<Context>) -> Self {
        let heartbeat = context.lock.lock().unwrap().new_heartbeat();

        Self {
            context,
            job_queue: JobQueue::default(),
            heartbeat,
            join_count: 0,
            shared_jobs: Default::default(),
        }
    }

    fn heartbeat_id(&self) -> usize {
        Arc::as_ptr(&self.heartbeat) as usize
    }

    fn return_job(&mut self, job_id: usize) -> Option<Box<dyn ExecuteJob<'s> + 's>> {
        let mut shared_jobs = self.shared_jobs.lock().unwrap();
        if shared_jobs
            .get(&self.heartbeat_id())
            .map(|(_, shared_job)| job_id == shared_job.id())
            .is_some()
        {
            if let Some((_, job)) = shared_jobs.remove(&self.heartbeat_id()) {
                return Some(job);
            }
        }

        None
    }

    fn wait_for_sent_job<T: Send>(&mut self, fut: &Future<T>) -> thread::Result<T> {
        // SAFETY:
        // For this `Job` to have crossed thread borders, it must have been
        // popped from the `JobQueue` and shared.
        while !fut.poll() {
            if let Some(job) = self.pop_earliest_shared_job() {
                // SAFETY:
                // Any `Job` that was shared between threads is waited upon
                // before the `JobStack` exits scope.
                unsafe {
                    job.execute(self);
                }
            } else {
                break;
            }
        }

        // SAFETY:
        // Any `Job` that was shared between threads is waited upon before the
        // `JobStack` exits scope.
        fut.wait()
    }

    #[cold]
    fn heartbeat(&mut self) {
        let mut lock = self.context.lock.lock().unwrap();

        let time = lock.time;
        let mut shared_jobs = self.shared_jobs.lock().unwrap();
        if let Entry::Vacant(e) = shared_jobs.entry(self.heartbeat_id()) {
            if let Some(job) = self.job_queue.pop_front() {
                e.insert((time, job));

                lock.time += 1;
                self.context.job_is_ready.notify_one();
            }
        }

        self.heartbeat.store(false, Ordering::Relaxed);
    }

    fn join_seq<A, B, RA, RB>(&mut self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce(&mut Self) -> RA + Send,
        B: FnOnce(&mut Self) -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let rb = b(self);
        let ra = a(self);

        (ra, rb)
    }

    fn join_heartbeat<A, B, RA, RB>(&mut self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce(&mut Self) -> RA + Send + 's,
        B: FnOnce(&mut Self) -> RB + Send + 's,
        RA: Send + 's,
        RB: Send + 's,
    {
        let a = move |scope: &mut Self| {
            if scope.heartbeat.load(Ordering::Relaxed) {
                scope.heartbeat();
            }

            a(scope)
        };

        let stack = JobStack::new(a);
        let job = Job::new(&stack);
        let fut = job.prepare();
        let job_id = job.id();

        self.job_queue.push_back(Box::new(job));

        let rb = b(self);

        if self.job_queue.pop_back().is_some() {
            // SAFETY:
            // Since the `job` was popped from the back of the queue, it cannot
            // take the closure out of the `JobStack` anymore.
            // `JobStack::take_once` is thus called only once.
            (unsafe { (stack.take_once())(self) }, rb)
        } else {
            if let Some(_job) = self.return_job(job_id) {
                return (unsafe { (stack.take_once())(self) }, rb);
            }

            let ra = match self.wait_for_sent_job(&fut) {
                Ok(val) => val,
                Err(e) => panic::resume_unwind(e),
            };

            (ra, rb)
        }
    }

    /// Runs `a` and `b` potentially in parallel on separate threads and
    /// returns the results.
    ///
    /// This variant skips checking for a heartbeat every 16 calls for improved
    /// performance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chili::{Scope, ThreadPool};
    /// let mut vals = [0; 2];
    /// let (mut left, mut right) = vals.split_at_mut(1);
    ///
    /// {
    ///    let (left_r, right_r) = (&mut left, &mut right);
    ///    ThreadPool::new().scope().join(|_| left_r[0] = 1, |_| right_r[0] = 1);
    /// }
    /// assert_eq!(vals, [1; 2]);
    /// ```
    pub fn join<A, B, RA, RB>(&mut self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce(&mut Self) -> RA + Send + 's,
        B: FnOnce(&mut Self) -> RB + Send + 's,
        RA: Send + 's,
        RB: Send + 's,
    {
        self.join_with_heartbeat_every::<64, _, _, _, _>(a, b)
    }

    /// Runs `a` and `b` potentially in parallel on separate threads and
    /// returns the results.
    ///
    /// This variant skips checking for a heartbeat every `TIMES - 1` calls for
    /// improved performance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chili::{Scope, ThreadPool};
    ///
    /// let mut vals = [0; 2];
    /// let (mut left, mut right) = vals.split_at_mut(1);
    /// let (left_r, right_r) = (&mut left, &mut right);
    ///
    /// // Skip checking 7/8 calls to join_with_heartbeat_every.
    /// {
    /// ThreadPool::new().scope().join_with_heartbeat_every::<8, _, _, _, _>(|_| left_r[0] = 1, |_| right_r[0] = 1);
    /// }
    ///
    /// assert_eq!(vals, [1; 2]);
    /// ```
    pub fn join_with_heartbeat_every<const TIMES: u8, A, B, RA, RB>(
        &mut self,
        a: A,
        b: B,
    ) -> (RA, RB)
    where
        A: FnOnce(&mut Self) -> RA + Send + 's,
        B: FnOnce(&mut Self) -> RB + Send + 's,
        RA: Send + 's,
        RB: Send + 's,
    {
        self.join_count = self.join_count.wrapping_add(1) % TIMES;

        if self.join_count == 0 || self.job_queue.len() < 3 {
            self.join_heartbeat(a, b)
        } else {
            self.join_seq(a, b)
        }
    }

    /// TODO
    ///
    ///
    /// # Examples
    pub fn pop_earliest_shared_job(&mut self) -> Option<Box<dyn ExecuteJob<'s> + 's>> {
        self.shared_jobs
            .lock()
            .unwrap()
            .pop_first()
            .map(|(_, (_, shared_job))| shared_job)
    }
}

/// `ThreadPool` configuration.
#[derive(Debug)]
pub struct Config {
    /// The number of threads or `None` to use
    /// `std::thread::available_parallelism`.
    pub thread_count: Option<NonZero<usize>>,
    /// The interval between heartbeats on any particular thread.
    pub heartbeat_interval: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            thread_count: None,
            heartbeat_interval: Duration::from_micros(100),
        }
    }
}

static GLOBAL_THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// A thread pool for running fork-join workloads.
#[derive(Debug)]
pub struct ThreadPool {
    context: Arc<Context>,
    worker_handles: Vec<JoinHandle<()>>,
    heartbeat_handle: Option<JoinHandle<()>>,
}

impl ThreadPool {
    /// Crates a new thread pool with default `Config`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chili::ThreadPool;
    /// let _tp = ThreadPool::new();
    /// ```
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    /// Creates a new thread pool with `config`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::{num::NonZero, time::Duration};
    /// # use chili::{Config, ThreadPool};
    /// let _tp = ThreadPool::with_config(Config {
    ///     thread_count: Some(NonZero::new(1).unwrap()),
    ///     heartbeat_interval: Duration::from_micros(50),
    /// });
    /// ```
    pub fn with_config(config: Config) -> Self {
        let thread_count = config
            .thread_count
            .or_else(|| thread::available_parallelism().ok())
            .map(|thread_count| thread_count.get() - 1)
            .unwrap_or_default();
        let worker_barrier = Arc::new(Barrier::new(thread_count + 1));

        let context = Arc::new(Context {
            lock: Mutex::new(LockContext::default()),
            job_is_ready: Condvar::new(),
            scope_created_from_thread_pool: Condvar::new(),
        });

        let worker_handles = (0..thread_count)
            .map(|_| {
                let context = context.clone();
                let barrier = worker_barrier.clone();
                thread::spawn(move || {
                    execute_worker(context, barrier);
                })
            })
            .collect();

        worker_barrier.wait();

        Self {
            context: context.clone(),
            worker_handles,
            heartbeat_handle: Some(thread::spawn(move || {
                execute_heartbeat(context, config.heartbeat_interval, thread_count);
            })),
        }
    }

    /// Sets the global thread pool to this one.
    ///
    /// The global thread pool can only be set once. Any subsequent call will
    /// return the thread pool back.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::{num::NonZero, time::Duration};
    /// # use chili::{Config, ThreadPool};
    /// ThreadPool::with_config(Config {
    ///     thread_count: Some(NonZero::new(1).unwrap()),
    ///     heartbeat_interval: Duration::from_micros(50),
    /// })
    /// .set_global()
    /// .unwrap();
    /// ```
    pub fn set_global(self) -> Result<(), Self> {
        GLOBAL_THREAD_POOL.set(self)
    }

    /// Returns the global thread pool.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chili::ThreadPool;
    ///
    /// let mut vals = [0; 2];
    /// let (left, right) = vals.split_at_mut(1);
    /// {
    /// let mut s = ThreadPool::global().scope();
    /// s.join(|_| left[0] = 1, |_| right[0] = 1);
    /// }
    /// assert_eq!(vals, [1; 2]);
    /// ```
    pub fn global() -> &'static ThreadPool {
        GLOBAL_THREAD_POOL.get_or_init(ThreadPool::new)
    }

    /// Returns a `Scope`d object that you can run fork-join workloads on.
    ///
    /// # Examples
    /// ```
    /// # use chili::ThreadPool;
    /// let mut vals = [0; 2];
    /// let (mut left, mut right) = vals.split_at_mut(1);
    /// let (left_r, right_r) = (&mut left, &mut right);
    /// {
    /// let tp = ThreadPool::new();
    /// let mut s = tp.scope();
    /// s.join(|_| left_r[0] = 1, |_| right_r[0] = 1);
    /// drop(s);
    /// drop(tp);
    /// }

    /// assert_eq!(vals, [1; 2]);
    /// ```
    pub fn scope(&self) -> Scope<'_> {
        Scope::new_from_thread_pool(self)
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.context
            .lock
            .lock()
            .expect("locking failed")
            .is_stopping = true;
        self.context.job_is_ready.notify_all();
        self.context.scope_created_from_thread_pool.notify_one();

        for handle in self.worker_handles.drain(..) {
            handle.join().unwrap();
        }

        if let Some(handle) = self.heartbeat_handle.take() {
            handle.join().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU8;

    use super::*;

    use thread::ThreadId;

    #[test]
    fn thread_pool_stops() {
        let _tp = ThreadPool::new();
    }

    #[test]
    fn thread_pool_with_one_thread() {
        let _tp = ThreadPool::with_config(Config {
            thread_count: Some(NonZero::new(1).unwrap()),
            ..Default::default()
        });
    }

    #[test]
    fn join_basic() {
        let threat_pool = ThreadPool::new();

        let mut a = 0;
        let mut b = 0;

        {
            let mut scope = threat_pool.scope();
            scope.join(|_| a += 1, |_| b += 1);
        }

        assert_eq!(a, 1);
        assert_eq!(b, 1);
    }

    #[test]
    fn join_long() {
        let threat_pool = ThreadPool::new();

        fn increment<'s>(s: &mut Scope<'s>, slice: &'s mut [u32]) {
            match slice.len() {
                0 => (),
                1 => slice[0] += 1,
                _ => {
                    let (head, tail) = slice.split_at_mut(1);

                    s.join(|_| head[0] += 1, |s| increment(s, tail));
                }
            }
        }

        let mut vals = [0; 1_024];

        increment(&mut threat_pool.scope(), &mut vals);

        assert_eq!(vals, [1; 1_024]);
    }

    #[test]
    fn join_very_long() {
        let threat_pool = ThreadPool::new();

        fn increment<'s>(s: &mut Scope<'s>, slice: &'s mut [u32]) {
            match slice.len() {
                0 => (),
                1 => slice[0] += 1,
                _ => {
                    let mid = slice.len() / 2;
                    let (left, right) = slice.split_at_mut(mid);

                    s.join(|s| increment(s, left), |s| increment(s, right));
                }
            }
        }

        let mut vals = vec![0; 1_024 * 1_024];

        increment(&mut threat_pool.scope(), &mut vals);

        assert_eq!(vals, vec![1; 1_024 * 1_024]);
    }

    #[test]
    fn join_wait() {
        let threat_pool = ThreadPool::with_config(Config {
            thread_count: Some(NonZero::new(2).unwrap()),
            heartbeat_interval: Duration::from_micros(1),
            ..Default::default()
        });

        fn increment<'s>(s: &mut Scope<'s>, slice: &'s mut [u32]) {
            match slice.len() {
                0 => (),
                1 => slice[0] += 1,
                _ => {
                    let (head, tail) = slice.split_at_mut(1);

                    s.join_with_heartbeat_every::<1, _, _, _, _>(
                        |_| {
                            thread::sleep(Duration::from_micros(10));
                            head[0] += 1;
                        },
                        |s| increment(s, tail),
                    );
                }
            }
        }

        let mut vals = [0; 10];

        increment(&mut threat_pool.scope(), &mut vals);

        assert_eq!(vals, [1; 10]);
    }

    #[test]
    #[should_panic(expected = "panicked across threads")]
    fn join_panic() {
        let threat_pool = ThreadPool::with_config(Config {
            thread_count: Some(NonZero::new(2).unwrap()),
            heartbeat_interval: Duration::from_micros(1),
        });

        if let Some(thread_count) = thread::available_parallelism().ok().map(NonZero::get) {
            if thread_count == 1 {
                // Pass test artificially when only one thread is available.
                panic!("panicked across threads");
            }
        }

        let threads_crossed = AtomicBool::new(false);

        fn increment<'s>(
            s: &mut Scope<'s>,
            slice: &'s mut [u32],
            id: ThreadId,
            threads_crossed: &'s AtomicBool,
        ) -> bool {
            match slice.len() {
                0 => (),
                1 => slice[0] += 1,
                _ => {
                    let (head, tail) = slice.split_at_mut(1);

                    s.join_with_heartbeat_every::<1, _, _, _, _>(
                        move |_| {
                            thread::sleep(Duration::from_micros(100));

                            if thread::current().id() != id {
                                threads_crossed.store(true, Ordering::Relaxed);
                                panic!("panicked across threads");
                            }

                            head[0] += 1;
                        },
                        move |s| increment(s, tail, id, threads_crossed),
                    );
                }
            }

            threads_crossed.load(Ordering::Relaxed)
        }

        let mut vals = [0; 10];

        let threads_crossed = increment(
            &mut threat_pool.scope(),
            &mut vals,
            thread::current().id(),
            &threads_crossed,
        );

        // Since there was no panic up to this point, this means that the
        // thread boundary has not been crossed.
        //
        // Check that the work was done and pass the test artificially.
        if !threads_crossed {
            assert_eq!(vals, [1; 10]);
            panic!("panicked across threads");
        }
    }

    #[test]
    fn concurrent_scopes() {
        const NUM_THREADS: u8 = 128;
        let threat_pool = ThreadPool::with_config(Config {
            thread_count: Some(NonZero::new(4).unwrap()),
            ..Default::default()
        });

        let a = AtomicU8::new(0);
        let b = AtomicU8::new(0);

        thread::scope(|s| {
            for _ in 0..NUM_THREADS {
                s.spawn(|| {
                    let mut scope = threat_pool.scope();
                    scope.join(
                        |_| a.fetch_add(1, Ordering::Relaxed),
                        |_| b.fetch_add(1, Ordering::Relaxed),
                    );
                });
            }
        });

        assert_eq!(a.load(Ordering::Relaxed), NUM_THREADS);
        assert_eq!(b.load(Ordering::Relaxed), NUM_THREADS);
    }
}
