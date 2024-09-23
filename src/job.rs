use std::{
    cell::{RefCell, UnsafeCell},
    collections::VecDeque,
    marker::PhantomData,
    mem::ManuallyDrop,
    panic::{self, AssertUnwindSafe},
    ptr::NonNull,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
    thread::{self, Thread},
};

use crate::Scope;

enum Poll {
    Pending,
    Ready,
    Locked,
}

#[derive(Debug, Default)]
pub struct Future<T> {
    state: AtomicU8,
    /// Can only be accessed if `state` is `Poll::Locked`.
    waiting_thread: UnsafeCell<Option<Thread>>,
    /// Can only be written if `state` is `Poll::Locked` and read if `state` is
    /// `Poll::Ready`.
    val: UnsafeCell<Option<Box<thread::Result<T>>>>,
}

impl<T> Future<T> {
    fn new() -> Self {
        Self {
            state: AtomicU8::default(),
            waiting_thread: UnsafeCell::new(None),
            val: UnsafeCell::new(None),
        }
    }

    pub fn poll(&self) -> bool {
        self.state.load(Ordering::Acquire) == Poll::Ready as u8
    }

    pub fn wait(&self) -> thread::Result<T> {
        loop {
            let result = self.state.compare_exchange(
                Poll::Pending as u8,
                Poll::Locked as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            );

            match result {
                Ok(_) => {
                    // SAFETY:
                    // Lock is acquired, only we are accessing `self.waiting_thread`.
                    unsafe { *self.waiting_thread.get() = Some(thread::current()) };

                    self.state.store(Poll::Pending as u8, Ordering::Release);

                    thread::park();

                    // Skip yielding after being woken up.
                    continue;
                }
                Err(state) if state == Poll::Ready as u8 => {
                    // SAFETY:
                    // `state` is `Poll::Ready` only after `Self::complete`
                    // releases the lock.
                    //
                    // Calling `Self::complete` when `state` is `Poll::Ready`
                    // cannot mutate `self.val`.
                    break unsafe { *self.val.get().as_mut().unwrap().take().unwrap() };
                }
                _ => (),
            }

            thread::yield_now();
        }
    }

    pub fn complete(&self, val: thread::Result<T>) {
        let val = Box::new(val);

        loop {
            let result = self.state.compare_exchange(
                Poll::Pending as u8,
                Poll::Locked as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            );

            match result {
                Ok(_) => break,
                Err(_) => thread::yield_now(),
            }
        }

        // SAFETY:
        // Lock is acquired, only we are accessing `self.val`.
        unsafe {
            *self.val.get() = Some(val);
        }

        // SAFETY:
        // Lock is acquired, only we are accessing `self.waiting_thread`.
        if let Some(thread) = unsafe { (*self.waiting_thread.get()).take() } {
            thread.unpark();
        }

        self.state.store(Poll::Ready as u8, Ordering::Release);
    }
}

pub struct JobStack<F> {
    /// All code paths should call either `Job::execute` or `Self::unwrap` to
    /// avoid a potential memory leak.
    f: UnsafeCell<ManuallyDrop<F>>,
}

impl<F> JobStack<F> {
    pub fn new(f: F) -> Self {
        Self {
            f: UnsafeCell::new(ManuallyDrop::new(f)),
        }
    }

    /// SAFETY:
    /// It should only be called once.
    pub unsafe fn take_once(&self) -> F {
        // SAFETY:
        // No `Job` has has been executed, therefore `self.f` has not yet been
        // `take`n.
        unsafe { ManuallyDrop::take(&mut *self.f.get()) }
    }
}

pub trait ExecuteJob<'s>: Send + Sync {
    unsafe fn execute(self: Box<Self>, scope: &mut Scope<'s>);

    fn id(&self) -> usize;
}

/// `Job` is only sent, not shared between threads.
///
/// When popped from the `JobQueue`, it gets copied before sending across
/// thread boundaries.
#[derive(Debug)]
pub struct Job<'s, F: FnOnce(&mut Scope<'s>) -> T + Send, T: Send> {
    stack: NonNull<JobStack<F>>,
    fut: RefCell<Option<Arc<Future<T>>>>,
    _phantom: PhantomData<&'s ()>,
}

impl<'s, F: FnOnce(&mut Scope<'s>) -> T + Send, T: Send> Job<'s, F, T> {
    pub fn new(stack: &JobStack<F>) -> Self
    where
        F: FnOnce(&mut Scope<'s>) -> T + Send,
        T: Send,
    {
        Self {
            stack: NonNull::from(stack).cast(),
            fut: RefCell::new(None),
            _phantom: PhantomData,
        }
    }

    pub fn prepare(&self) -> Arc<Future<T>> {
        let f = Arc::new(Future::new());
        self.fut.replace(Some(f.clone()));

        f
    }
}

impl<'s, F: FnOnce(&mut Scope<'s>) -> T + Send, T: Send> ExecuteJob<'s> for Job<'s, F, T> {
    /// SAFETY:
    /// It should only be called while the `JobStack` it was created with is
    /// still alive and after being popped from a `JobQueue`.
    unsafe fn execute(self: Box<Self>, scope: &mut Scope<'s>) {
        // SAFETY:
        // The `stack` is still alive.
        let stack: &JobStack<F> = unsafe { self.stack.cast().as_ref() };
        // SAFETY:
        // This is the first call to `take_once` since `Job::execute`
        // (the only place where this harness is called) is called only
        // after the job has been popped.
        let f = unsafe { stack.take_once() };

        self.fut
            .borrow()
            .as_ref()
            .unwrap()
            .complete(panic::catch_unwind(AssertUnwindSafe(|| f(scope))));
    }

    fn id(&self) -> usize {
        self.stack.as_ptr() as usize
    }
}

// SAFETY:
// The job's `stack` will only be accessed after acquiring a lock (in
// `Future`), while `prev` and `fut_or_next` are never accessed after being
// sent across threads.
unsafe impl<'s, F: FnOnce(&mut Scope<'s>) -> T + Send, T: Send> Send for Job<'s, F, T> {}

unsafe impl<'s, F: FnOnce(&mut Scope<'s>) -> T + Send, T: Send> Sync for Job<'s, F, T> {}

#[derive(Default)]
pub struct JobQueue<'s>(VecDeque<Box<dyn ExecuteJob<'s> + 's>>);

impl<'s> JobQueue<'s> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn push_back(&mut self, job: Box<dyn ExecuteJob<'s> + 's>) {
        self.0.push_back(job);
    }

    pub fn pop_back(&mut self) -> Option<Box<dyn ExecuteJob<'s> + 's>> {
        self.0.pop_back()
    }

    pub fn pop_front(&mut self) -> Option<Box<dyn ExecuteJob<'s> + 's>> {
        let job = self.0.pop_front()?;

        Some(job)
    }
}
