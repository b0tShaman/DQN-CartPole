package src.DQN;

import java.awt.*;
import javax.swing.*;

public class CartPole extends JPanel implements Runnable {

    private double theta = Math.PI / 2;

    private double x = Math.PI / 2;

    private int length;

    public CartPole(double length) {
        this.length = (int)(length * 1000);
        setDoubleBuffered(true);
    }

    public void setState( Environment_CartPole.State state){
        this.theta = state.theta;
        this.x = state.x;
    }

    @Override
    public void paint(Graphics g) {
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, getWidth(), getHeight());
        g.setColor(Color.BLACK);
        int anchorX = Math.round((float)x * 100) + (getWidth() / 2);
        int anchorY = getHeight() -(10) ;
        int ballX = anchorX + (int) (Math.sin(theta) * length);
        int ballY = anchorY - (int) (Math.cos(theta) * length) ;

        g.drawLine(anchorX, anchorY, ballX, ballY);
        g.fillRect(anchorX-25 , anchorY- 10 , 50, 20);
        g.fillOval(ballX - 7, ballY - 7, 14, 14);
    }

    public void run() {
        while(true) {
            repaint();
        }
    }

    @Override
    public Dimension getPreferredSize() {
        return new Dimension(2 * length + 50, length / 2 * 3);
    }

}
