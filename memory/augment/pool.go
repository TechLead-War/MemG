package augment

import "sync"

// Pool is a fixed-worker pool with a bounded job queue for augmentation work.
type Pool struct {
	queue chan func()
	wg    sync.WaitGroup
	done  chan struct{}
	once  sync.Once
}

// NewPool creates a pool with the given number of fixed workers and a queue
// capacity of workers*4. Workers pull from the queue; Submit returns false
// if the queue is full instead of blocking.
func NewPool(workers int) *Pool {
	queueSize := workers * 4
	p := &Pool{
		queue: make(chan func(), queueSize),
		done:  make(chan struct{}),
	}
	p.wg.Add(workers)
	for i := 0; i < workers; i++ {
		go p.worker()
	}
	return p
}

func (p *Pool) worker() {
	defer p.wg.Done()
	for fn := range p.queue {
		fn()
	}
}

// Submit schedules fn for execution. Returns true if the job was accepted,
// false if the queue is full. Never blocks.
func (p *Pool) Submit(fn func()) bool {
	select {
	case <-p.done:
		return false
	default:
	}
	select {
	case p.queue <- fn:
		return true
	default:
		return false
	}
}

// Shutdown signals the pool to stop accepting work, drains the queue,
// and waits for all workers to finish.
func (p *Pool) Shutdown() {
	p.once.Do(func() {
		close(p.done)
		close(p.queue)
	})
	p.wg.Wait()
}
