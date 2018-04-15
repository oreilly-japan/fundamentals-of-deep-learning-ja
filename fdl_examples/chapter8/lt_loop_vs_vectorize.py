import tensorflow as tf
import numpy as np

def Lt_loop(L, wwt, p, N):
    L_t = tf.zeros([N,N], tf.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            _mask = np.zeros([N,N], np.float32);
            _mask[i,j] = 1.0
            mask = tf.convert_to_tensor(_mask)
        
            link_t = (1 - wwt[i] - wwt[j]) * L[i,j] + wwt[i] * p[j]
            L_t += mask * link_t

    return L_t


def Lt(L, wwt, p, N):
    """
    returns the updated link matrix given the previous one along with
    the updated write weightings and the previous precedence vector
    """
    def pairwise_add(v):
        """
        returns the matrix of pairwe-adding the elements of v to
        themselves
        """
        n = v.get_shape().as_list()[0]
        # a NxN matrix of duplicates of u along the columns
        V = tf.concat([v] * n, 1)  
        return V + tf.transpose(V)

    # expand dimensions of wwt and p to make
    # matmul behave as outer product
    wwt = tf.expand_dims(wwt, 1)
    p = tf.expand_dims(p, 0)

    I = tf.constant(np.identity(N, dtype=np.float32))
    return ((1 - pairwise_add(wwt)) * L + tf.matmul(wwt, p)) * (1 - I)
    
    
if __name__ == '__main__':
    N=10

    L_np = np.random.random((N, N))
    wwt_np = np.random.random(N)
    p_np = np.random.random(N)

    with tf.Graph().as_default() as graph:
        L = tf.Variable(L_np, dtype=tf.float32)
        wwt = tf.Variable(wwt_np, dtype=tf.float32)
        p = tf.Variable(p_np, dtype=tf.float32)
        
        lt_loop = Lt_loop(L, wwt, p, N)
        print('#Nodes of loop version: ', len(graph.as_graph_def().node))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            lt_loop_ = sess.run(lt_loop)
        
    with tf.Graph().as_default() as graph:
        L = tf.Variable(L_np, dtype=tf.float32)
        wwt = tf.Variable(wwt_np, dtype=tf.float32)
        p = tf.Variable(p_np, dtype=tf.float32)
        
        lt = Lt(L, wwt, p, N)
        print('#Nodes of vectorized version: ', len(graph.as_graph_def().node))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            lt_ = sess.run(lt)
            
    # Check the results are same.
    print(np.isclose(lt_loop_, lt_).all())